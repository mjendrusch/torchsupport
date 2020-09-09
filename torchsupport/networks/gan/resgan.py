import torch
import torch.nn as nn
import torch.nn.functional as func

class ResGeneratorBlock(nn.Module):
  def __init__(self, in_size, out_size, kernel_size=3, activation=None, weight=None):
    super().__init__()
    self.activation = activation or nn.LeakyReLU(0.2)
    hidden_size = min(in_size, out_size)
    self.blocks = nn.Sequential(
      self.activation,
      nn.Conv2d(in_size, hidden_size, kernel_size, padding=kernel_size // 2),
      self.activation,
      nn.Conv2d(hidden_size, out_size, kernel_size, padding=kernel_size // 2)
    )
    self.skip = nn.Conv2d(in_size, out_size, 1, bias=False)
    self.weight = weight
    if weight is None:
      self.weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

  def forward(self, inputs):
    out = self.weight * self.blocks(inputs) + self.skip(inputs)
    return out

class ResGeneratorBackbone(nn.Module):
  def __init__(self, in_size=100, base_channels=64, channel_factors=None,
               kernel_size=3, activation=None, weight=None):
    super().__init__()
    self.initial = nn.Linear(in_size, 4 * 4 * base_channels * channel_factors[0])
    self.blocks = nn.ModuleList([
      ResGeneratorBlock(
        in_factor * base_channels,
        out_factor * base_channels,
        kernel_size=kernel_size,
        activation=activation,
        weight=weight
      )
      for in_factor, out_factor in zip(
        channel_factors[:-1], channel_factors[1:]
      )
    ])

  def forward(self, inputs):
    out = self.initial(inputs).view(inputs.size(0), -1, 4, 4)
    for block in self.blocks:
      out = func.interpolate(out, scale_factor=2)
      out = block(out)
    return out

class ResDiscriminatorBackbone(nn.Module):
  def __init__(self, in_size=3, base_channels=64, channel_factors=None,
               kernel_size=3, activation=None, weight=None):
    super().__init__()
    self.preprocess = nn.Conv2d(
      in_size, base_channels * channel_factors[0], 3, padding=1
    )
    self.blocks = nn.ModuleList([
      ResGeneratorBlock(
        in_factor * base_channels,
        out_factor * base_channels,
        kernel_size=kernel_size,
        activation=activation,
        weight=weight
      )
      for in_factor, out_factor in zip(
        channel_factors[:-1], channel_factors[1:]
      )
    ])

  def forward(self, inputs):
    out = self.preprocess(inputs)
    for block in self.blocks:
      out = block(out)
      out = func.avg_pool2d(out, 2)
    out = func.adaptive_avg_pool2d(out, 1)
    return out
