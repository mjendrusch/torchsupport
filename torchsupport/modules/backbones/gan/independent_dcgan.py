r"""This module implements a parametric histoGAN / DCGAN architecture suitable for generating histology images."""
import random

import torch
import torch.nn as nn
import torch.nn.functional as func

def _generator_layer(in_channels, out_channels, kernel_size, activation,
                     expand_first=False):
  return nn.Sequential(
    nn.ConvTranspose2d(
      in_channels,
      out_channels,
      kernel_size,
      padding=kernel_size - 1,
      output_padding=0,
      stride=2
    ),
    nn.BatchNorm2d(out_channels),
    activation
  )

def _discriminator_layer(in_channels, out_channels, kernel_size, activation):
  return nn.Sequential(
    nn.Conv2d(
      in_channels, out_channels, kernel_size,
      padding=kernel_size // 2, stride=2
    ),
    nn.BatchNorm2d(out_channels),
    activation
  )

class LocalDCGANGenerator(nn.Module):
  def __init__(self, in_size=100, base_channels=64, channel_factors=None,
               kernel_size=4, activation=None):
    super().__init__()
    self.activation = activation or nn.ReLU()
    self.project_in = nn.Conv2d(
      in_size, base_channels * channel_factors[0], 1,
      bias=False
    )
    self.blocks = nn.ModuleList([
      _generator_layer(
        in_channels=in_factor * base_channels,
        out_channels=out_factor * base_channels,
        kernel_size=kernel_size,
        activation=self.activation
      )
      for in_factor, out_factor in zip(
        channel_factors[:-1], channel_factors[1:]
      )
    ])

  def forward(self, inputs):
    out = self.project_in(inputs)
    for block in self.blocks:
      out = block(out)
    return out

class LevelDCGANGenerator(nn.Module):
  def __init__(self, base_channels=128, cond_size=512, local_size=10,
               channel_factors=None, target_size=128, activation=None,
               kernel_size=4):
    super().__init__()
    self.cond_size = cond_size
    self.local_size = local_size
    self.target_size = target_size
    self.channel_factors = channel_factors
    self.base_channels = base_channels
    self.kernel_size = kernel_size

    self.levels = nn.ModuleList([
      LocalDCGANGenerator(
        self.cond_size + self.local_size + 3,
        base_channels=base_channels,
        channel_factors=cf,
        activation=activation,
        kernel_size=kernel_size
      )
      for cf in channel_factors
    ])

    self.rgb = nn.ModuleList([
      nn.Conv2d(cf[-1] * base_channels, 3, 1)
      for cf in channel_factors
    ])

  def forward(self, condition):
    scale = 2 ** (len(self.channel_factors[0]) - 1)
    size = self.kernel_size + self.target_size // scale
    cond = condition[:, :, None, None].expand(
      *condition.shape, size, size
    )

    local_latent = torch.randn(
      cond.size(0), self.local_size, *cond.shape[2:],
      device=cond.device
    )
    rgb = torch.zeros(
      cond.size(0), 3, *cond.shape[2:],
      device=cond.device
    )
    out = torch.cat((cond, local_latent, rgb), dim=1)
    conditions = [None]
    results = []
    for idx, (block, to_rgb) in enumerate(zip(self.levels, self.rgb)):
      out = block(out)
      rgb = to_rgb(out).tanh()
      off = (rgb.size(-1) - self.target_size) // 2
      rgb = rgb[:, :, off:-off, off:-off]
      out = out[:, :, off:-off, off:-off]
      results.append(rgb)
      if idx < len(self.levels) - 1:
        scale = 2 ** (len(self.channel_factors[idx + 1]) - 1)
        size = self.kernel_size + self.target_size // scale
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
        cond = condition[:, :, None, None].expand(
          *condition.shape, size, size
        )
        local_latent = torch.randn(
          cond.size(0), self.local_size, *cond.shape[2:],
          device=cond.device
        )
        out = torch.cat((cond, local_latent, rgb), dim=1)
    conditions[0] = torch.zeros_like(results[0])
    return results, conditions
