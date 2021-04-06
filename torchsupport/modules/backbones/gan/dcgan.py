r"""This module implements a parametric histoGAN / DCGAN architecture suitable for generating histology images."""

import torch
import torch.nn as nn
import torch.nn.functional as func

def _generator_layer(position, in_channels, out_channels, kernel_size, activation,
                     expand_first=False):
  expand_first = expand_first and position
  return nn.Sequential(
    nn.ConvTranspose2d(
      in_channels,
      out_channels,
      kernel_size,
      padding=kernel_size // 2 if expand_first else 0,
      output_padding=1 if expand_first else 0,
      stride=2 if expand_first else 1
    ),
    nn.BatchNorm2d(out_channels),
    activation
  )

def _discriminator_layer(in_channels, out_channels, kernel_size, activation):
  return nn.Sequential(
    nn.Conv2d(
      in_channels, out_channels, kernel_size,
      padding=(kernel_size - 1) // 2, stride=2
    ),
    nn.BatchNorm2d(out_channels),
    activation
  )

class DCGANGenerator(nn.Module):
  def __init__(self, in_size=100, base_channels=64, channel_factors=None,
               kernel_size=5, initial_size=4, activation=None):
    super().__init__()
    expand_first = (kernel_size - 1 + initial_size) == 2 * initial_size
    self.initial_size = initial_size
    self.activation = activation or nn.ReLU()
    self.initial = nn.Linear(
      in_size,
      self.initial_size ** 2 * base_channels * channel_factors[0]
    )
    self.blocks = nn.ModuleList([
      _generator_layer(
        position=idx,
        in_channels=in_factor * base_channels,
        out_channels=out_factor * base_channels,
        kernel_size=kernel_size,
        activation=self.activation,
        expand_first=expand_first
      )
      for idx, (in_factor, out_factor) in enumerate(zip(
        channel_factors[:-1], channel_factors[1:]
      ))
    ])

  def forward(self, inputs):
    out = self.initial(inputs).view(
      inputs.size(0), -1,
      self.initial_size,
      self.initial_size
    )
    for block in self.blocks:
      out = block(out)
    return out

class DCGANDiscriminator(nn.Module):
  def __init__(self, in_size=3, base_channels=64, channel_factors=None,
               kernel_size=5, activation=None):
    super().__init__()
    self.activation = activation or nn.ReLU()
    self.preprocess = _discriminator_layer(
      in_channels=in_size,
      out_channels=base_channels * channel_factors[0],
      kernel_size=kernel_size,
      activation=self.activation
    )
    self.blocks = nn.ModuleList([
      _discriminator_layer(
        in_channels=in_factor * base_channels,
        out_channels=out_factor * base_channels,
        kernel_size=kernel_size,
        activation=self.activation
      )
      for idx, (in_factor, out_factor) in enumerate(zip(
        channel_factors[:-1], channel_factors[1:]
      ))
    ])

  def forward(self, inputs):
    out = self.preprocess(inputs)
    for block in self.blocks:
      out = block(out)
    return out

class DCGANPatchDiscriminator(DCGANDiscriminator):
  def __init__(self, in_size=3, base_channels=64, channel_factors=None,
               kernel_size=5, activation=None):
    super().__init__(
      in_size=in_size,
      base_channels=base_channels,
      channel_factors=channel_factors,
      kernel_size=kernel_size,
      activation=activation
    )
    self.predict = nn.Conv2d(base_channels * channel_factors[-1], 1, 1)

  def forward(self, inputs):
    out = super().forward(inputs)
    out = self.predict(out)
    out = out.reshape(out.size(0), 1, -1).permute(2, 0, 1).reshape(-1, out.size(1))
    return out
