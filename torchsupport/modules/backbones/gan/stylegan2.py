import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules import ReZero, lr_equal
from torchsupport.modules.weights import WeightDemodulation

class Mapping(nn.Module):
  def __init__(self, latent_size=512, output_size=512,
               hidden_size=128, depth=8, lr_scale=0.01,
               activation=func.relu):
    super().__init__()
    self.depth = depth
    self.activation = activation
    self.project_in = lr_equal(nn.Linear(latent_size, hidden_size), lr_scale=lr_scale)
    self.project_out = lr_equal(nn.Linear(hidden_size, output_size), lr_scale=lr_scale)
    self.blocks = nn.ModuleList([
      lr_equal(nn.Linear(hidden_size, hidden_size), lr_scale=lr_scale)
      for idx in range(depth)
    ])

  def normalize_latent(self, latent, condition=None):
    latent = latent / latent.norm(dim=1, keepdim=True)
    if condition:
      condition = condition / condition.norm(dim=1, keepdim=True)
      latent = torch.cat((latent, condition), dim=1)
    return latent

  def forward(self, latent, condition=None):
    out = self.normalize_latent(latent, condition=condition)
    for block in self.blocks:
      out = self.activation(block(out))
    return out

class ResidualMapping(Mapping):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.zeros = nn.ModuleList([
      ReZero()
      for idx in range(self.depth)
    ])

  def forward(self, latent, condition=None):
    out = self.normalize_latent(latent, condition=condition)
    for zero, block in zip(self.blocks, self.zeros):
      out = zero(out, self.activation(block(out)))
    return out

class StyleGAN2ConvBlock(nn.Module):
  def __init__(self, in_size, out_size, cond_size=512, depth=2, activation=None):
    super().__init__()
    self.activation = activation or nn.LeakyReLU(0.2)
    self.conds = nn.ModuleList([
      lr_equal(nn.Linear(cond_size, in_size, bias=False))
      for idx in range(depth)
    ])
    self.blocks = nn.ModuleList([
      WeightDemodulation(
        nn.Conv2d(in_size, in_size, 3, padding=1)
      )
      for idx in range(depth - 1)
    ] + [
      WeightDemodulation(
        nn.Conv2d(in_size, out_size, 3, padding=1)
      )
    ])
    self.noise_scale = nn.ParameterList([
      torch.randn(in_size)
      for idx in range(depth - 1)
    ] + [
      torch.randn(out_size)
    ])
    self.rgb = nn.Conv2d(out_size, 3, 1)

  def forward(self, inputs, condition):
    out = inputs
    for block, cond, scale in zip(
        self.blocks, self.conds, self.scales
    ):
      out = block(out, cond(condition))
      noise = torch.randn_like(out)
      noise = noise.view(*noise.shape[:2], -1) * scale[:, None]
      noise = noise.view(*out.shape)
      out = out + noise
    rgb = self.rgb(out)
    return out, rgb

class StyleGAN2GeneratorBackbone(nn.Module):
  def __init__(self, base_channels=128, cond_size=512, channel_factors=None,
               base_size=4, block_depth=2, activation=None):
    super().__init__()
    self.base_map = nn.Parameter(torch.randn(
      1, base_channels * channel_factors[0], base_size, base_size
    ))
    self.base_rgb = torch.zeros(1, 3, base_size, base_size)
    self.blocks = nn.ModuleList([
      StyleGAN2ConvBlock(
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

  def forward(self, condition):
    out = self.base_map.repeat_interleave(condition.size(0), dim=0)
    rgb = self.base_rgb.repeat_interleave(condition.size(0), dim=0)
    for idx, block in enumerate(self.blocks):
      out, rgb_update = block(out, condition)
      rgb = rgb + rgb_update
      if idx < len(self.blocks) - 1:
        out = func.interpolate(out, scale_factor=2)
        rgb = func.interpolate(rgb, scale_factor=2)
    return rgb

class StyleGAN2DiscriminatorBlock(nn.Module):
  def __init__(self, in_size, out_size, activation=None):
    super().__init__()
    self.activation = activation or nn.LeakyReLU(0.2)
    self.blocks = nn.ModuleList([
      lr_equal(nn.Conv2d(in_size, out_size, 3, padding=1)),
      lr_equal(nn.Conv2d(out_size, out_size, 3, padding=1))
    ])
    self.zero = ReZero(out_size)

  def forward(self, inputs):
    out = inputs
    for block in self.blocks:
      out = self.activation(block(out))
    if out.size(1) > inputs.size(1):
      difference = out.size(1) - inputs.size(1)
      padding = torch.zeros(
        inputs.size(0), difference, *inputs.shape[2:]
      )
      inputs = torch.cat((inputs, padding), dim=1)
    return self.zero(inputs, out)

class StyleGAN2DiscriminatorBackbone(nn.Module):
  def __init__(self, in_size, base_channels=16, channel_factors=None,
               activation=None):
    super().__init__()
    self.project = lr_equal(nn.Conv2d(
      in_size, base_channels * channel_factors[0], 1, bias=False
    ))
    self.blocks = nn.ModuleList([
      StyleGAN2DiscriminatorBlock(
        base_channels * in_factor,
        base_channels * out_factor,
        activation=activation
      )
      for idx, (in_factor, out_factor) in enumerate(zip(
        channel_factors[:-1], channel_factors[1:]
      ))
    ])

  def forward(self, inputs):
    out = self.project(inputs)
    for idx, block in self.blocks:
      out = block(out)
      if idx < len(self.blocks) - 1:
        out = func.interpolate(
          out,
          scale_factor=0.5,
          mode="bilinear"
        )
    return out
