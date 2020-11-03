import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.rezero import ReZero

class NestedModule(nn.Module):
  def __init__(self, hole):
    super().__init__()
    self.hole = hole

  def enter(self, inputs, *args, **kwargs):
    raise NotImplementedError("Abstract.")

  def exit(self, inputs, skip, *args, **kwargs):
    raise NotImplementedError("Abstract.")

  def forward(self, inputs, *args, **kwargs):
    out, skip = self.enter(inputs, *args, **kwargs)
    out = self.hole(out, *args, **kwargs)
    out = self.exit(out, skip, *args, **kwargs)
    return out

class NestedWrapper(NestedModule):
  def __init__(self, hole, left=None, right=None):
    super().__init__(hole)
    self.left = left or nn.Identity()
    self.right = right or nn.Identity()

  def enter(self, inputs, *args, **kwargs):
    out = self.left(inputs, *args, **kwargs)
    skip = out
    return out, skip

  def exit(self, inputs, skip, *args, **kwargs):
    inputs = torch.cat((inputs, skip), dim=1)
    return self.right(inputs, *args, **kwargs)

class LightResUNetBlock(NestedModule):
  def __init__(self, hole, in_size=64, out_size=64,
               hidden_size=64, kernel_size=3, dilation=1,
               padding=None, depth=3, downscale=2,
               cond_size=None, activation=None):
    super().__init__(hole)
    padding = padding or (kernel_size // 2 * dilation)
    self.activation = activation or func.relu
    self.cond = None
    if cond_size:
      self.cond = nn.Linear(cond_size, hidden_size)
    self.downscale = downscale
    self.into_preprocess = nn.Conv2d(in_size, hidden_size, 1)
    self.into_postprocess = nn.Conv2d(hidden_size, out_size, 1)
    self.into_bn = nn.InstanceNorm2d(in_size)
    self.into_blocks = nn.ModuleList([
      nn.Conv2d(
        hidden_size,
        hidden_size,
        kernel_size,
        padding=padding
      )
      for idx in range(depth)
    ])
    self.into_zeros = nn.ModuleList([
      ReZero(hidden_size)
      for idx in range(depth)
    ])
    self.outof_preprocess = nn.Conv2d(
      out_size + hidden_size, hidden_size, 1
    )
    self.outof_bn = nn.InstanceNorm2d(out_size + hidden_size)
    self.outof_postprocess = nn.Conv2d(hidden_size, in_size, 1)
    self.outof_blocks = nn.ModuleList([
      nn.Conv2d(
        hidden_size,
        hidden_size,
        kernel_size,
        padding=padding
      )
      for idx in range(depth)
    ])
    self.outof_zeros = nn.ModuleList([
      ReZero(hidden_size)
      for idx in range(depth)
    ])

  def enter(self, inputs, *args, **kwargs):
    out = self.activation(self.into_preprocess(self.into_bn(inputs)))
    for zero, block in zip(self.into_zeros, self.into_blocks):
      this_out = out
      if self.cond:
        cond = torch.cat(args, dim=1)
        cond = self.cond(cond)[:, :, None, None]
        this_out = this_out + cond
      res = func.dropout(
        self.activation(block(this_out)),
        0.1, training=self.training
      )
      out = zero(out, res)
    skip = out
    out = self.into_postprocess(out)
    if self.downscale != 1:
      out = func.avg_pool2d(out, self.downscale)
    return out, skip

  def exit(self, inputs, skip, *args, **kwargs):
    if self.downscale != 1:
      inputs = func.interpolate(inputs, scale_factor=self.downscale, mode="bilinear")
    out = self.activation(self.outof_preprocess(self.outof_bn(torch.cat((inputs, skip), dim=1))))
    for zero, block in zip(self.outof_zeros, self.outof_blocks):
      this_out = out
      if self.cond:
        cond = torch.cat(args, dim=1)
        cond = self.cond(cond)[:, :, None, None]
        this_out = this_out + cond
      res = func.dropout(
        self.activation(block(this_out)),
        0.1, training=self.training
      )
      out = zero(out, res)
    out = self.outof_postprocess(out)
    return out

def nest_sequential(*args):
  block = args[-1]
  for item in reversed(args[:-1]):
    item.hole = block
    block = item
  return block

class Ignore(nn.Module):
  def forward(self, x, *args, **kwargs):
    return x

class IgnoreArgs(nn.Module):
  def __init__(self, net):
    super().__init__()
    self.net = net

  def forward(self, x, *args, **kwargs):
    return self.net(x)

class LightUNetBackbone(nn.Module):
  def __init__(self, hole=None, size_factors=None,
               kernel_size=None, base_size=64, **kwargs):
    super().__init__()
    kernel_size = kernel_size or 3
    self.blocks = nest_sequential(*([
      LightResUNetBlock(
        None, in_size=base_size * in_factor,
        kernel_size=kernel_size,
        out_size=base_size * out_factor,
        **kwargs
      )
      for in_factor, out_factor in zip(
        size_factors[:-1], size_factors[1:]
      )
    ] + [
      hole or Ignore()
    ]))

  def forward(self, inputs, *args, **kwargs):
    return self.blocks(inputs, *args, **kwargs)


class ResSubblock(nn.Module):
  def __init__(self, in_size=64, out_size=64, cond_size=None,
               kernel_size=3, dilation=1, padding=None,
               activation=None, norm=None):
    super().__init__()
    padding = padding or (kernel_size // 2 * dilation)
    self.cond = None
    self.norm = norm or (lambda x: nn.Identity())
    self.activation = activation or func.relu
    if cond_size:
      self.cond = nn.Linear(cond_size, out_size)
    self.blocks = nn.ModuleList([
      nn.Conv2d(in_size, out_size, kernel_size, dilation, padding),
      nn.Conv2d(out_size, out_size, kernel_size, dilation, padding)
    ])
    self.project = nn.Conv2d(in_size, out_size, 1, bias=False)
    self.zero = ReZero(out_size)
    self.norms = nn.ModuleList([
      self.norm(in_size),
      self.norm(out_size)
    ])

  def forward(self, inputs, cond=None):
    out = inputs
    out = self.blocks[0](self.activation(self.norms[0](out)))  
    if cond is not None:
      out = out + self.cond(cond)[:, :, None, None]
    out = func.dropout(
      self.activation(self.norms[1](out)),
      0.1, training=self.training
    )
    out = self.blocks[1](out)
    out = self.zero(self.project(inputs), out)
    return out

class ResUNetBlock(NestedModule):
  def __init__(self, hole, in_size=64, out_size=64,
               hidden_size=64, kernel_size=3, dilation=1,
               padding=None, depth=2, downscale=2,
               cond_size=None, activation=None, norm=None):
    super().__init__(hole)
    self.activation = activation or func.relu
    self.downscale = downscale
    self.into_blocks = nn.ModuleList([
      ResSubblock(
        in_size, in_size, cond_size=cond_size,
        kernel_size=kernel_size, dilation=dilation,
        padding=padding, activation=activation,
        norm=norm
      )
      for idx in range(depth - 1)
    ] + [
      ResSubblock(
        in_size, out_size, cond_size=cond_size,
        kernel_size=kernel_size, dilation=dilation,
        padding=padding, activation=activation,
        norm=norm
      )
    ])
    self.outof_blocks = nn.ModuleList([
      ResSubblock(
        2 * out_size, in_size, cond_size=cond_size,
        kernel_size=kernel_size, dilation=dilation,
        padding=padding, activation=activation,
        norm=norm
      )
    ] + [
      ResSubblock(
        in_size, in_size, cond_size=cond_size,
        kernel_size=kernel_size, dilation=dilation,
        padding=padding, activation=activation,
        norm=norm
      )
      for idx in range(depth - 1)
    ])

  def enter(self, inputs, cond=None):
    out = inputs
    for block in self.into_blocks:
      out = block(out, cond=cond)
    skip = out
    if self.downscale != 1:
      out = func.avg_pool2d(out, self.downscale)
    return out, skip

  def exit(self, inputs, skip, cond=None):
    if self.downscale != 1:
      inputs = func.interpolate(inputs, scale_factor=self.downscale, mode="nearest")
    out = torch.cat((inputs, skip), dim=1)
    for block in self.outof_blocks:
      out = block(out, cond=cond)
    return out

class UNetBackbone(nn.Module):
  def __init__(self, hole=None, size_factors=None, kernel_size=None, base_size=64, **kwargs):
    super().__init__()
    kernel_size = kernel_size or 3
    self.blocks = nest_sequential(*([
      ResUNetBlock(
        None, in_size=base_size * in_factor,
        kernel_size=kernel_size,
        out_size=base_size * out_factor,
        **kwargs
      )
      for in_factor, out_factor in zip(
        size_factors[:-1], size_factors[1:]
      )
    ] + [
      hole or Ignore()
    ]))

  def forward(self, inputs, *args, **kwargs):
    return self.blocks(inputs, *args, **kwargs)
