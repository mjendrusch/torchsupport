import torch
import torch.nn as nn
import torch.nn.functional as func

import torchsupport.modules.normalization as tsn
from torchsupport.modules.attention import NonLocal

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
    self.convs = nn.ModuleList([
      nn.Conv2d(in_size, in_size, 3),
      nn.Conv2d(in_size, out_size, 3)
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

class BigGANBlock(nn.Module):
  def __init__(self, in_size, out_size,
               hidden_size=None, upsample=1,
               normalization=tsn.AdaptiveInstanceNorm,
               activation=func.relu_):
    super(BigGANBlock, self).__init__()
    if hidden_size is None:
      hidden_size = in_size
    self.in_size = in_size
    self.out_size = out_size
    self.upsample = upsample
    self.bn = nn.ModuleList([
      normalization(in_size),
      normalization(hidden_size),
      normalization(hidden_size),
      normalization(hidden_size)
    ])
    self.blocks = nn.ModuleList([
      nn.Conv2d(in_size, hidden_size, 1),
      nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
      nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
      nn.Conv2d(hidden_size, out_size, 1)
    ])
    self.activation = activation

  def forward(self, inputs, latent):
    skip = inputs[:, :self.in_size]
    skip = func.interpolate(skip, scale_factor=self.upsample)

    out = inputs
    for idx, (bn, block) in enumerate(zip(self.bn, self.blocks)):
      out = self.activation(bn(out, latent))
      if idx == 1:
        out = func.interpolate(out, scale_factor=self.upsample)
      out = block(out)

    return out + skip

class BigGANDiscriminatorBlock(nn.Module):
  def __init__(self, in_size, out_size,
               hidden_size=None, downsample=2,
               activation=func.relu_):
    super(BigGANDiscriminatorBlock, self).__init__()
    if hidden_size is None:
      hidden_size = out_size
    self.in_size = in_size
    self.out_size = out_size
    self.hidden_size = hidden_size
    self.downsample = downsample
    self.blocks = nn.ModuleList([
      nn.Conv2d(in_size, hidden_size, 1),
      nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
      nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
      nn.Conv2d(hidden_size, out_size, 1)
    ])
    self.activation = activation
    self.project = nn.Conv2d(in_size, out_size - in_size, 1)
  
  def forward(self, inputs):
    skip = func.avg_pool2d(inputs, self.downsample)
    missing = self.project(skip)
    skip = torch.cat((skip, missing), dim=1)

    out = inputs
    for idx, block in enumerate(self.blocks):
      out = self.activation(out)
      if idx == len(self.blocks) - 1:
        out = func.avg_pool2d(inputs, self.downsample)
      out = block(out)

    return skip + out
