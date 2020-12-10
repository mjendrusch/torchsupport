import random

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.weights import lr_equal, WeightDemodulation

class IndependentStyleGAN2Block(nn.Module):
  def __init__(self, in_size, out_size, cond_size=512, depth=2, activation=None):
    super().__init__()
    self.activation = activation or nn.LeakyReLU(0.2)
    self.conds = nn.ModuleList([
      lr_equal(nn.Linear(cond_size, in_size, bias=False))
      for idx in range(depth)
    ])
    self.blocks = nn.ModuleList([
      WeightDemodulation(
        nn.Conv2d(in_size, in_size, 1)
      )
      for idx in range(depth - 1)
    ] + [
      WeightDemodulation(
        nn.Conv2d(in_size, out_size, 1)
      )
    ])
    self.noise_scale = nn.ParameterList([
      nn.Parameter(torch.zeros(in_size, requires_grad=True))
      for idx in range(depth - 1)
    ] + [
      nn.Parameter(torch.zeros(out_size, requires_grad=True))
    ])
    self.rgb = lr_equal(nn.Conv2d(out_size, 3, 1))

  def forward(self, inputs, condition):
    out = inputs
    for block, cond, scale in zip(
        self.blocks, self.conds, self.noise_scale
    ):
      out = block(out, cond(condition))
      noise = torch.randn_like(out)
      noise = noise.view(*noise.shape[:2], -1) * scale[:, None]
      noise = noise.view(*out.shape)
      # out = out + noise
    rgb = self.rgb(out)
    return out, rgb

class IndependentStyleGAN2GeneratorBackbone(nn.Module):
  def __init__(self, base_channels=128, cond_size=512, channel_factors=None,
               block_depth=2, activation=None):
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
        activation=activation
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
