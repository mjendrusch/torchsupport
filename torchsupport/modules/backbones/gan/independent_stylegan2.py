import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules import ReZero
from torchsupport.modules.weights import lr_equal, WeightDemodulation

class IndependentStyleGAN2Block(nn.Module):
  def __init__(self, in_size, out_size, cond_size=512, depth=2,
               add_noise=False, kernel_size=1, activation=None):
    super().__init__()
    self.add_noise = add_noise
    self.activation = activation or nn.LeakyReLU(0.2)
    self.conds = nn.ModuleList([
      lr_equal(nn.Linear(cond_size, in_size, bias=True))
      for idx in range(depth)
    ])
    with torch.no_grad():
      for cond in self.conds:
        cond.module.bias.zero_().add_(1.0)
    self.blocks = nn.ModuleList([
      WeightDemodulation(
        nn.Conv2d(in_size, in_size, kernel_size)
      )
      for idx in range(depth - 1)
    ] + [
      WeightDemodulation(
        nn.Conv2d(in_size, out_size, kernel_size)
      )
    ])
    self._noise_scale = []
    for idx in range(depth):
      self._noise_scale.append(idx)
      size = in_size if idx < depth - 1 else out_size
      setattr(self, f"noise_scale_{idx}", nn.Parameter(torch.zeros(1, requires_grad=True)))
    self.rgb = lr_equal(nn.Conv2d(out_size, 3, 1))

  @property
  def noise_scale(self):
    return [
      getattr(self, f"noise_scale_{scale}")
      for scale in self._noise_scale
    ]

  def forward(self, inputs, condition):
    out = inputs
    for block, cond, scale in zip(
        self.blocks, self.conds, self.noise_scale
    ):
      out = block(out, cond(condition))
      if self.add_noise:
        noise = torch.randn_like(out)
        noise = noise.view(*noise.shape[:2], -1) * scale[:, None]
        noise = noise.view(*out.shape)
        out = out + noise
    rgb = self.rgb(out)
    return out, rgb

class ProjectCoordinates(nn.Module):
  def __init__(self, cond_size=512, in_size=2, out_size=512, scale=50.0):
    super().__init__()
    self.project_coordinates = nn.Linear(
      cond_size, in_size * out_size
    )
    self.bias = nn.Linear(cond_size, out_size)
    self.scale = scale

  def forward(self, coordinates, condition):
    shape = coordinates.shape
    coordinates = coordinates.permute(0, 2, 3, 1).reshape(shape[0], -1, 2)
    projector = self.project_coordinates(condition).view(condition.size(0), 2, -1)
    projector = self.scale * projector
    out = ((coordinates @ projector) + self.bias(condition)[:, None, :]).sin()
    out = out.view(shape[0], *shape[2:], out.shape[-1]).permute(0, 3, 1, 2).contiguous()
    return out

class LocalMapping(nn.Module):
  def __init__(self, in_size=512, cond_size=512, hidden_size=128,
               depth=3, lr_scale=0.01, activation=None):
    super().__init__()
    self.project_in = lr_equal(nn.Conv2d(in_size, hidden_size, 1), lr_scale=lr_scale)
    self.project_out = lr_equal(nn.Conv2d(hidden_size, cond_size, 1), lr_scale=lr_scale)
    self.blocks = nn.ModuleList([
      lr_equal(nn.Conv2d(hidden_size, hidden_size, 1), lr_scale=lr_scale)
      for idx in range(depth - 2)
    ])
    self.activation = activation or nn.LeakyReLU(0.2)

  def forward(self, inputs):
    out = inputs / inputs.norm(dim=1, keepdim=True)
    out = self.activation(self.project_in(out))
    for block in self.blocks:
      out = self.activation(block(out))
    return self.project_out(out)

class LocalStyleGAN2GeneratorBackbone(nn.Module):
  def __init__(self, base_channels=128, cond_size=512, channel_factors=None,
               block_depth=2, scale=50.0, activation=None, add_noise=True):
    super().__init__()
    self.channel_factors = channel_factors
    self.base_channels = base_channels
    self.block_depth = block_depth
    self.project = ProjectCoordinates(
      cond_size=cond_size,
      in_size=2,
      out_size=cond_size,
      scale=scale
    )
    self.project_local = LocalMapping(cond_size=cond_size)
    self.combine = lr_equal(nn.Conv2d(
      2 * cond_size,
      base_channels * channel_factors[0],
      1
    ))

    self.blocks = nn.ModuleList([
      IndependentStyleGAN2Block(
        in_factor * base_channels,
        out_factor * base_channels,
        cond_size=cond_size,
        depth=block_depth,
        activation=activation,
        kernel_size=3,
        add_noise=add_noise
      )
      for idx, (in_factor, out_factor) in enumerate(zip(
        channel_factors[:-1], channel_factors[1:]
      ))
    ])

    self.zeros = nn.ModuleList([
      ReZero(3)
      for idx, (in_factor, out_factor) in enumerate(zip(
        channel_factors[:-1], channel_factors[1:]
      ))
    ])

  def input_size(self, target_size):
    steps = len(self.blocks)
    pad = self.block_depth
    size = target_size
    for idx in range(steps):
      size = (size + 2 * pad) / 2
    size = int(math.ceil(2 * size))
    return size

  def forward(self, coordinates, condition, mix=None):
    rgb = 0.0
    out = self.project(coordinates, condition)
    local_latent = self.project_local(torch.randn_like(out))
    out = torch.cat((torch.randn_like(out), local_latent), dim=1)
    out = self.combine(out)
    cond = condition
    off = self.block_depth
    for idx, block in enumerate(self.blocks):
      if mix is not None:
        cond = random.choice((condition, mix))
      out, rgb_update = block(out, cond)
      if torch.is_tensor(rgb):
        rgb = rgb[:, :, off:-off, off:-off]
        rgb = rgb + rgb_update
      else:
        rgb = rgb_update
      if idx < len(self.blocks) - 1:
        out = func.interpolate(out, scale_factor=2, mode="bilinear")
        rgb = func.interpolate(rgb, scale_factor=2, mode="bilinear")
    return rgb

class LevelStyleGAN2GeneratorBlock(nn.Module):
  def __init__(self, in_channels, base_channels=128, cond_size=512,
               channel_factors=None, block_depth=2, activation=None,
               add_noise=True):
    super().__init__()
    self.block_depth = block_depth
    self.activation = activation or nn.LeakyReLU(0.2)
    self.project_in = lr_equal(nn.Conv2d(
      in_channels, base_channels * channel_factors[0], 1, bias=False
    ))
    self.blocks = nn.ModuleList([
      IndependentStyleGAN2Block(
        in_factor * base_channels,
        out_factor * base_channels,
        cond_size=cond_size,
        depth=block_depth,
        activation=activation,
        kernel_size=3,
        add_noise=add_noise
      )
      for idx, (in_factor, out_factor) in enumerate(zip(
        channel_factors[:-1], channel_factors[1:]
      ))
    ])

  def forward(self, rgb, features, condition, mix=None):
    out = self.activation(self.project_in(features))
    cond = condition
    off = self.block_depth
    for idx, block in enumerate(self.blocks):
      if mix is not None:
        cond = random.choice((condition, mix))
      out, rgb_update = block(out, cond)
      if torch.is_tensor(rgb):
        rgb = rgb[:, :, off:-off, off:-off]
        rgb = rgb + rgb_update
      else:
        rgb = rgb_update
      if idx < len(self.blocks) - 1:
        out = func.interpolate(out, scale_factor=2, mode="bilinear")
        rgb = func.interpolate(rgb, scale_factor=2, mode="bilinear")
    return out, rgb

class LevelStyleGAN2GeneratorBackbone(nn.Module):
  def __init__(self, base_channels=128, cond_size=512, channel_factors=None,
               block_depth=2, scale=50.0, target_size=128, activation=None,
               add_noise=True):
    super().__init__()
    self.target_size = target_size
    self.channel_factors = channel_factors
    self.base_channels = base_channels
    self.block_depth = block_depth
    self.project = ProjectCoordinates(
      cond_size=cond_size,
      in_size=2,
      out_size=cond_size,
      scale=scale
    )
    self.project_local = LocalMapping(cond_size=cond_size)

    self.levels = nn.ModuleList([
      LevelStyleGAN2GeneratorBlock(
        2 * cond_size,
        base_channels=base_channels,
        cond_size=cond_size,
        channel_factors=cf,
        block_depth=block_depth,
        activation=activation,
        add_noise=add_noise
      )
      for cf in channel_factors
    ])

    self.projections = nn.ModuleList([
      lr_equal(nn.Conv2d(cf[-1] * base_channels + 3, cond_size, 1, bias=False))
      for cf in channel_factors
    ])

    coordinate_size = self.input_size(0)
    self.inputs = nn.Parameter(
      torch.randn(cond_size, coordinate_size, coordinate_size, requires_grad=True)
    )

  def input_size(self, level):
    steps = len(self.levels[level].blocks)
    pad = self.block_depth
    size = self.target_size
    for _ in range(steps):
      size = (size + 2 * pad) / 2
    size = int(math.ceil(2 * size))
    print(size)
    return size

  def forward(self, coordinates, condition, mix=None):
    rgb = 0.0
    out = self.project(coordinates, condition)
    out = torch.randn_like(out)
    # out = self.inputs[None].expand(coordinates.size(0), *self.inputs.shape)
    local_latent = self.project_local(torch.randn_like(out))
    out = torch.cat((out, local_latent), dim=1)
    conditions = [None]
    results = []
    for idx, (block, project) in enumerate(zip(self.levels, self.projections)):
      out, rgb_update = block(rgb, out, condition, mix=mix)
      if torch.is_tensor(rgb):
        rgb = func.interpolate(rgb, size=rgb_update.size(-1), mode="bilinear")
      rgb = rgb + rgb_update
      off = (rgb.size(-1) - self.target_size) // 2
      rgb_target = rgb
      if off > 0:
        rgb_target = rgb_target[:, :, off:-off, off:-off]
      results.append(rgb_target)
      if idx < len(self.levels) - 1:
        size = self.input_size(idx + 1)
        xs = random.randrange(out.size(-1) - size)
        ys = random.randrange(out.size(-1) - size)
        xe = xs + size
        ye = ys + size
        out = out[:, :, xs:xe, ys:ye]
        rgb = rgb[:, :, xs:xe, ys:ye]
        scale = 2 ** (len(self.channel_factors[idx + 1]) - 1)
        rescaled = func.interpolate(rgb, scale_factor=scale, mode="bilinear")
        off = (rescaled.size(-1) - self.target_size) // 2
        conditions.append(rescaled[:, :, off:-off, off:-off])
        out = project(torch.cat((out, rgb), dim=1))
        out = torch.cat((torch.randn_like(out), out), dim=1)
    first_scale = 2 ** len(self.levels[0].blocks)
    print(results[0].shape, first_scale)
    first = func.interpolate(results[0], scale_factor=1 / 4, mode="bilinear")
    first = func.interpolate(first, scale_factor=4, mode="bilinear")
    conditions[0] = torch.zeros_like(first)
    return results, conditions

class IndependentStyleGAN2GeneratorBackbone(nn.Module):
  def __init__(self, base_channels=128, cond_size=512, channel_factors=None,
               block_depth=2, activation=None, add_noise=False):
    super().__init__()
    # TODO: refactor once Hypernetworks are implemented.
    self.project_coordinates = nn.Linear(
      cond_size, 2 * base_channels * channel_factors[0]
    )
    self.bias = nn.Linear(cond_size, base_channels * channel_factors[0])
    # self.project = lr_equal(nn.Conv2d(2, base_channels * channel_factors[0], 1))
    self.blocks = nn.ModuleList([
      IndependentStyleGAN2Block(
        in_factor * base_channels,
        out_factor * base_channels,
        cond_size=cond_size,
        depth=block_depth,
        activation=activation,
        add_noise=add_noise
      )
      for idx, (in_factor, out_factor) in enumerate(zip(
        channel_factors[:-1], channel_factors[1:]
      ))
    ])

  def forward(self, coordinates, condition, mix=None):
    rgb = 0.0
    shape = coordinates.shape
    coordinates = coordinates.permute(0, 2, 3, 1).reshape(shape[0], -1, 2)
    projector = self.project_coordinates(condition).view(condition.size(0), 2, -1)
    out = ((coordinates @ projector) + self.bias(condition)[:, None, :]).sin()
    out = out.view(shape[0], *shape[2:], out.shape[-1]).permute(0, 3, 1, 2).contiguous()
    cond = condition
    for idx, block in enumerate(self.blocks):
      if mix is not None:
        cond = random.choice((condition, mix))
      out, rgb_update = block(out, cond)
      rgb = rgb + rgb_update
    return rgb

class IndependentScaleBlock(IndependentStyleGAN2GeneratorBackbone):
  def __init__(self, in_size=None, base_channels=128, cond_size=512,
               channel_factors=None, block_depth=2, activation=None):
    super().__init__(
      base_channels=base_channels,
      cond_size=cond_size,
      channel_factors=channel_factors,
      block_depth=block_depth,
      activation=activation
    )
    if in_size is None:
      in_size = base_channels * channel_factors[0]
    self.project_in = lr_equal(nn.Conv2d(
      in_size + base_channels * channel_factors[0],
      base_channels * channel_factors[0],
      1
    ))

  def forward(self, rgb, inputs, coordinates, condition, mix=None):
    shape = coordinates.shape
    coordinates = coordinates.permute(0, 2, 3, 1).reshape(shape[0], -1, 2)
    projector = self.project_coordinates(condition).view(condition.size(0), 2, -1)
    out = ((coordinates @ projector) + self.bias(condition)[:, None, :]).sin()
    out = out.view(shape[0], *shape[2:], out.shape[-1]).permute(0, 3, 1, 2).contiguous()
    if inputs is not None:
      out = torch.cat((inputs, out), dim=1)
    if rgb is None:
      rgb = 0.0
    out = self.project_in(out)
    cond = condition
    for idx, block in enumerate(self.blocks):
      if mix is not None:
        cond = random.choice((condition, mix))
      out, rgb_update = block(out, cond)
      rgb = rgb + rgb_update
    return rgb, out

class LevelIndependentStyleGAN2GeneratorBackbone(nn.Module):
  def __init__(self, base_channels=128, cond_size=512, channel_factors=None,
               block_depth=2, scales=None, scale_factor=2, activation=None):
    super().__init__()
    self.scales = scales
    self.scale_factor = scale_factor
    self.blocks = nn.ModuleList([
      IndependentScaleBlock(
        base_channels=base_channels,
        channel_factors=factor_set,
        block_depth=block_depth,
        activation=activation
      )
      for factor_set in channel_factors
    ])

  def select(self, coordinates, rgb, out, shape, scale):
    x = random.randrange(shape[0] - scale)
    y = random.randrange(shape[1] - scale)
    c_slice = coordinates[:, x:x + scale, y:y + scale]
    r_slice = rgb[:, x:x + scale, y:y + scale]
    o_slice = out[:, x:x + scale, y:y + scale]
    return c_slice, r_slice, o_slice

  def forward(self, coordinates, condition, mix=None):
    coordinate_shape = coordinates.shape[2:]
    out = None
    rgb = None
    results = []
    conditions = [None]
    for idx, block in enumerate(self.blocks):
      rgb, out = block(rgb, out, coordinates, condition, mix=mix)
      if idx in self.scales:
        results.append(rgb)
        coordinates, rgb, out = self.select(
          coordinates, rgb, out, coordinate_shape, self.scales[idx]
        )
        conditions.append(func.interpolate(
          rgb, scale_factor=self.scale_factor
        ))
      if idx < len(self.blocks) - 1:
        coordinates = func.interpolate(
          coordinates, scale_factor=self.scale_factor
        )
        rgb = func.interpolate(
          rgb, scale_factor=self.scale_factor
        )
        out = func.interpolate(
          out, scale_factor=self.scale_factor
        )
    results.append(rgb)
    return results, conditions

class MultiscaleIndependentStyleGAN2GeneratorBackbone(IndependentStyleGAN2GeneratorBackbone):
  def __init__(self, *args, scales=None, inject_coordinates=False, base_channels=128,
               channel_factors=None, cond_size=512, **kwargs):
    super().__init__(
      *args, base_channels=base_channels,
      channel_factors=channel_factors,
      cond_size=cond_size, **kwargs
    )
    self.scales = scales
    self.inject_coordinates = inject_coordinates
    if inject_coordinates:
      self.coordmap = nn.ModuleDict({
        str(scale): nn.ModuleList([
          nn.Linear(cond_size, 2 * base_channels * channel_factors[scale + 1]),
          nn.Linear(cond_size, base_channels * channel_factors[scale + 1])
        ])
        for scale in self.scales
      })

  def project(self, proj, bias, condition, coordinates, shape):
    projector = proj(condition).view(condition.size(0), 2, -1)
    out = ((coordinates @ projector) + bias(condition)[:, None, :]).sin()
    out = out.view(shape[0], *shape[2:], out.shape[-1]).permute(0, 3, 1, 2).contiguous()
    return out

  def forward(self, coordinates, condition, mix=None):
    rgb = 0.0
    shape = coordinates.shape
    coordinates = coordinates.permute(0, 2, 3, 1).reshape(shape[0], -1, 2)
    out = self.project(
      self.project_coordinates, self.bias,
      condition, coordinates, shape
    )
    cond = condition
    results = []
    for idx, block in enumerate(self.blocks):
      if mix is not None:
        cond = random.choice((condition, mix))
      out, rgb_update = block(out, cond)
      rgb = rgb + rgb_update
      if idx in self.scales:
        results.append(rgb)
        if self.inject_coordinates:
          coord = self.project(
            *self.coordmap[str(idx)], cond,
            coordinates, shape
          )
          out = out + coord
    results.append(rgb)
    return results
