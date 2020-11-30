import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils.spectral_norm import spectral_norm

class BigGANGeneratorResBlock(nn.Module):
  def __init__(self, in_size, out_size, cond_size,
               up=2, activation=func.relu_,
               normalization=nn.BatchNorm2d):
    r"""Single residual block of a BigGAN generator.

    Args:
      in_size (int): number of input features.
      out_size (int): number of output features.
      cond_size (int): number of condition features.
      up (int or None): upsampling scale. If None, does
        not perform upsampling.
      activation (function): nonlinear activation function.
        Defaults to ReLU.
      normalization (function): normalization function.
        Defaults to BatchNorm2d.
    """
    super().__init__()
    self.up = up
    self.activation = activation
    self.in_mod = spectral_norm(nn.Linear(cond_size, 2 * in_size, bias=False))
    self.out_mod = spectral_norm(nn.Linear(cond_size, 2 * in_size, bias=False))
    self.in_norm = normalization(in_size, affine=False)
    self.out_norm = normalization(in_size, affine=False)
    self.in_conv = spectral_norm(nn.Conv2d(in_size, in_size, 3, padding=1, bias=False))
    self.out_conv = spectral_norm(nn.Conv2d(in_size, out_size, 3, padding=1, bias=False))
    self.skip_conv = spectral_norm(nn.Conv2d(in_size, out_size, 1, bias=False))

  def forward(self, inputs, condition):
    skip = inputs
    if self.up:
      skip = func.interpolate(skip, scale_factor=self.up, mode="bilinear")
    skip = self.skip_conv(skip)

    scale, bias = self.in_mod(condition).view(*condition.shape, 1, 1).chunk(2, dim=1)
    out = self.activation((1 + scale) * self.in_norm(inputs) + bias)
    out = self.in_conv(out)
    scale, bias = self.out_mod(condition).view(*condition.shape, 1, 1).chunk(2, dim=1)
    out = self.activation((1 + scale) * self.out_norm(inputs) + bias)
    out = self.out_conv(out)
    return skip + out

class BigGANDiscriminatorResBlock(nn.Module):
  def __init__(self, in_size, out_size, down=2,
               resize_first=False, activation=func.relu_):
    r"""Single residual block of a BigGAN discriminator.

    Args:
      in_size (int): number of input features.
      out_size (int): number of output features.
      down (int or None): downsampling scale. If None, does
        not perform downsampling.
      activation (function): nonlinear activation function.
        Defaults to ReLU.
      resize_first (bool): downsample before the first
        convolution?
    """
    super().__init__()
    self.resize_first = resize_first
    self.down = down
    self.activation = activation
    self.in_conv = spectral_norm(nn.Conv2d(in_size, out_size, 3, padding=1, bias=True))
    self.out_conv = spectral_norm(nn.Conv2d(out_size, out_size, 3, padding=1, bias=True))
    self.skip_conv = spectral_norm(nn.Conv2d(in_size, out_size, 1, bias=False))

  def forward(self, inputs):
    if self.resize_first and self.down:
      inputs = func.avg_pool2d(inputs, self.down)
    skip = self.skip_conv(inputs)

    out = self.in_conv(self.activation(inputs))
    out = self.out_conv(self.activation(out))

    if self.down and not self.resize_first:
      skip = func.avg_pool2d(out, self.down)
      out = func.avg_pool2d(out, self.down)

    return skip + out

class BigGANGeneratorBackbone(nn.Module):
  def __init__(self, base_channels=128, cond_size=128,
               channel_factors=None, upsampling_factors=None,
               **kwargs):
    r"""Parametric BigGAN generator backbone.

    Args:
      base_channels (int): base number of channels per feature map.
      cond_size (int): number of condition features.
      channel_factors (list): list of multiplicative factors of the
        base channel number. Specifies the size of feature maps
        throughout the generator backbone.
      upsampling_factors (list): list of upsampling scales for each
        block in the generator backbone.
    """
    super().__init__()
    self.blocks = nn.ModuleList([
      BigGANGeneratorResBlock(
        base_channels * first,
        base_channels * second,
        cond_size=cond_size, up=up,
        **kwargs
      )
      for first, second, up in zip(
        channel_factors[:-1],
        channel_factors[1:],
        upsampling_factors
      )
    ])

  def forward(self, inputs, conditions):
    if not isinstance(conditions, (list, tuple)):
      conditions = [conditions for idx in range(len(self.blocks))]
    out = inputs
    for block, condition in zip(self.blocks, conditions):
      out = block(out, condition)
    return out

class BigGANDiscriminatorBackbone(nn.Module):
  def __init__(self, base_channels=128, channel_factors=None,
               downsampling_factors=None, resize_first=True,
               **kwargs):
    r"""Parametric BigGAN discriminator backbone.

    Args:
      base_channels (int): base number of channels per feature map.
      channel_factors (list): list of multiplicative factors of the
        base channel number. Specifies the size of feature maps
        throughout the discriminator backbone.
      downsampling_factors (list): list of downsampling scales for each
        block in the discriminator backbone.
    """
    super().__init__()
    self.blocks = nn.ModuleList([
      BigGANDiscriminatorResBlock(
        base_channels * first,
        base_channels * second,
        resize_first=(idx == 0 and resize_first),
        down=down, **kwargs
      )
      for idx, (first, second, down) in enumerate(zip(
        channel_factors[:-1],
        channel_factors[1:],
        downsampling_factors
      ))
    ])

  def forward(self, inputs):
    out = inputs
    for block in self.blocks:
      out = block(out)
    return out
