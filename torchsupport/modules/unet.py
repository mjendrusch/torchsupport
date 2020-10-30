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

class ResUNetBlock(NestedModule):
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
    out = self.activation(self.into_preprocess(inputs))
    if self.cond:
      cond = torch.cat(args, dim=1)
      cond = self.cond(cond)[:, :, None, None]
      out = out + cond
    for zero, block in zip(self.into_zeros, self.into_blocks):
      res = func.dropout(
        self.activation(block(out)),
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
    out = self.activation(self.outof_preprocess(torch.cat((inputs, skip), dim=1)))
    if self.cond:
      cond = torch.cat(args, dim=1)
      cond = self.cond(cond)[:, :, None, None]
      out = out + cond
    for zero, block in zip(self.outof_zeros, self.outof_blocks):
      res = func.dropout(
        self.activation(block(out)),
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

class UNetBackbone(nn.Module):
  def __init__(self, size_factors=None, kernel_size=None, base_size=64, **kwargs):
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
      nn.Identity()
    ]))

  def forward(self, inputs, *args, **kwargs):
    return self.blocks(inputs, *args, **kwargs)
