import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.normalization import AdaptiveBatchNorm

class ResBlock(nn.Module):
  def __init__(self, size, ada_size):
    super().__init__()
    self.bn = AdaptiveBatchNorm(size, ada_size)
    self.convs = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(size, size, 3, dilation=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(size, size, 3, dilation=2, padding=2)
    )

  def forward(self, inputs, condition):
    out = self.bn(inputs, condition)
    out = self.convs(out)
    return out + inputs

class PoolBlock(nn.Module):
  def __init__(self, size, ada_size, depth=3):
    super().__init__()
    self.bn = nn.ModuleList([
      AdaptiveBatchNorm(size, ada_size)
      for idx in range(depth)
    ])
    self.blocks = nn.ModuleList([
      nn.Conv2d(size, size, 3, padding=1)
      for idx in range(depth)
    ])

  def forward(self, inputs, condition):
    out = inputs
    for bn, block in zip(self.bn, self.blocks):
      inner = func.avg_pool2d(bn(out, condition), 5, stride=1, padding=2)
      inner = block(inner)
      out = out + inner
    return out

class RefineBlock(nn.Module):
  def __init__(self, size, ada_size):
    super().__init__()
    self.res_low = nn.ModuleList([
      ResBlock(size, ada_size)
      for idx in range(2)
    ])
    self.res_high = nn.ModuleList([
      ResBlock(size, ada_size)
      for idx in range(2)
    ])
    self.low = nn.Conv2d(size, size, 3, padding=1)
    self.high = nn.Conv2d(size, size, 3, padding=1)
    self.pool = PoolBlock(size, ada_size)

  def forward(self, low, high, condition):
    for block in self.res_low:
      low = block(low, condition)
    low = func.interpolate(low, scale_factor=2, mode="bilinear")
    for block in self.res_high:
      high = block(high, condition)
    out = self.low(low) + self.high(high)
    out = self.pool(out, condition)
    return out
