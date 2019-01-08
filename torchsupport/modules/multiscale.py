import torch
import torch.nn as nn
import torch.nn.functional as func

class DilationCascade(nn.Module):
  def __init__(self, channels, kernel_size, levels=[1,2,4,8], merger=None):
    """
    Performs a series of dilated convolutions on a single input.
    Args:
      channels (int): number of input and output channels.
      kernel_size (int): convolutional kernel size.
      levels (list int): list of convolutional scales.
    """
    super(DilationCascade, self).__init__()
    self.merger = None
    self.levels = nn.ModuleList([
      nn.Conv2d(channels, channels, kernel_size,
                dilation=level, padding=(kernel_size // 2) * level)
      for level in levels
    ])

  def forward(self, input):
    if self.merger != None
      outputs = []
      out = input
      for level in self.levels:
        out = level(out)
        outputs.append(out)
      return self.merger(outputs)
    else:
      out = input
      for level in self.levels:
        out = level(out)
      return out

class DilatedMultigrid(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, levels=[0,1,2,4],
               merger=lambda x: torch.cat(x, dim=1)):
    """
    Dilated multi-grid convolution block.
    Args:
      in_channels (int): number of input channels.
      out_channels (int): number of output channels.
      kernel_size (int): convolutional kernel size.
      levels (list int): list of convolutional scales.
      merger (callable): procedure for merging multiple scale features.
    """
    super(DilatedMultigrid, self).__init__()
    self.merger = merger
    self.levels = nn.ModuleList([
      nn.Conv2d(in_channels, out_channels, kernel_size,
                dilation=level, padding=(kernel_size // 2) * level)
      if level != 0 else nn.Conv2d(in_channels, out_channels, 1)
      for level in levels
    ])

  def forward(self, input):
    outputs = []
    for level in self.levels:
      outputs.append(level(input))
    return self.merger(outputs)

def DilatedPyramid(channels, kernel_size, levels=[1,2,4,8],
                   merger=lambda x: torch.cat(x, dim=1)):
  """
  Pyramid construction version of `DilationCascade`. See `DilationCascade`.
  """
  return DilationCascade(channels, kernel_size, levels=levels, merger=merger)

class PoolingMultigrid(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, levels=[3,5,7,9],
               merger=lambda x: torch.cat(x, dim=1)):
    """
    Pooled multi-grid convolution block.
    Args:
      in_channels (int): number of input channels.
      out_channels (int): number of output channels.
      kernel_size (int): convolutional kernel size.
      levels (list int): list of convolutional scales.
      merger (callable): procedure for merging multiple scale features.
    """
    super(PoolingMultigrid, self).__init__()
    self.merger = merger
    self.levels = nn.ModuleList([
      nn.Sequential(
        nn.MaxPool2d(level, stride=1, padding=level // 2),
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
      )
      if level != 0 else nn.Conv2d(in_channels, out_channels, 1)
      for level in levels
    ])

  def forward(self, input):
    outputs = []
    for level in self.levels:
      outputs.append(level(input))
    return self.merger(outputs)

class PoolingPyramid(nn.Module):
  def __init__(self, channels, kernel_size, pooling_size, depth=4,
               merger=lambda x: torch.cat(x, dim=1)):
    """
    Iterative pooling image pyramid construction.
    Args:
      channels (int): number of input channels.
      kernel_size (int): convolutional kernel size.
      pooling_size (int): pooling kernel size.
      depth (int): number of pyramid layers.
      merger (callable): procedure for merging pyramid features.
    """
    super(PoolingPyramid, self).__init__()
    self.merger = merger
    self.levels = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2),
        nn.MaxPool2d(pooling_size)
      )
      for _ in range(depth)
    ])
    self.post_levels = nn.ModuleList([
      nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
      for _ in range(depth)
    ])

  def forward(self, input):
    outputs = []
    pass_through = input
    for idx, level in enumerate(self.levels):
      pass_through = level(pass_through)
      outputs.append(self.post_levels(pass_through))
    return self.merger(outputs)
