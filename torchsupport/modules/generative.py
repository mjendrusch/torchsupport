import torch
import torch.nn as nn
import torch.nn.functional as func

import torchsupport.modules.normalization as tsn

class UpsampleBlock(nn.Module):
  def __init__(self, in_size, out_size,
               size=None, upsample=2,
               activation=func.elu):
    super(UpsampleBlock, self).__init__()
    self.is_first = False
    self.size = size
    if size is not None:
      self.is_first = True
      total_size = torch.Size(size).numel()
      self.input = nn.Linear(in_size, out_size * total_size)
    self.pixnorm = tsn.PixelNorm()
    self.convs = nn.Module([
      nn.Conv2d(in_channels, in_channels, 3),
      nn.Conv2d(in_channels, out_channels, 3)
    ])
    self.activation = activation
    self.upsample = upsample

  def forward(self, inputs):
    out = inputs
    if self.is_first:
      out = self.activation(self.input(out))
      out = out.view(out.size(0), -1, *self.size)
      out = self.pixnorm(out)
    else:
      out = func.interpolate(out, scale_factor=self.upsample)
      out = self.activation(self.convs[0](out))
      out = self.pixnorm(out)
    out = self.activation(self.convs[1](out))
    return self.pixnorm(out)

class StyleGANBlock(nn.Module):
  def __init__(self, in_size, out_size, ada_size,
               size=None, upsample=2, activation=func.elu):
    super(StyleGANBlock, self).__init__()
    self.upsample = upsample
    self.register_parameter(
      "noise_0",
      nn.Parameter(torch.randn(1, in_size, 1, 1))
    )
    self.register_parameter(
      "noise_1",
      nn.Parameter(torch.randn(1, out_size, 1, 1))
    )

    self.is_first = False
    if size is not None:
      self.is_first = True
      self.register_parameter(
        "start_map",
        nn.Parameter(torch.randn(1, in_size, *size))
      )
    self.convs = nn.ModuleList([
      nn.Conv2d(in_size, in_size, 3, padding=1),
      nn.Conv2d(in_size, out_size, 3, padding=1)
    ])
    self.adas = nn.ModuleList([
      tsn.AdaptiveInstanceNorm(in_size, ada_size),
      tsn.AdaptiveInstanceNorm(out_size, ada_size)
    ])
    self.activation = activation

  def forward(self, inputs, latent, noise=None):
    out = inputs
    if self.is_first:
      out = self.start_map
      out = out.expand(latent.size(0), *out.shape[1:])
    else:
      out = func.interpolate(out, scale_factor=self.upsample)
      out = self.activation(self.convs[0](out))
    out = out + torch.randn_like(out) * self.noise_0
    out = self.adas[0](out, latent)
    out = self.activation(self.convs[1](out))
    out = out + torch.randn_like(out) * self.noise_1
    out = self.adas[1](out, latent)
    return out
