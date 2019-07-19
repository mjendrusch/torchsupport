import torch
import torch.nn as nn
import torch.nn.functional as func

class PixelNorm(nn.Module):
  def __init__(self, eps=1e-16, p=2):
    super(PixelNorm, self).__init__()
    self.eps = eps
    self.p = 2

  def forward(self, inputs):
    return inputs / torch.norm(inputs, dim=1, keepdim=True, p=self.p)

class AdaptiveInstanceNorm(nn.Module):
  def __init__(self, in_size, ada_size):
    super(AdaptiveInstanceNorm, self).__init__()
    self.scale = nn.Linear(ada_size, in_size)
    self.bias = nn.Linear(ada_size, in_size)

  def forward(self, inputs, style):
    in_view = inputs.view(inputs.size(0), inputs.size(1), 1, 1, -1)
    mean = in_view.mean(dim=-1)
    std = in_view.std(dim=-1)
    scale = self.scale(style).view(style.size(0), -1, 1, 1)
    bias = self.bias(style).view(style.size(0), -1, 1, 1)
    print(inputs.shape, mean.shape, std.shape, scale.shape, bias.shape)
    return scale * (inputs - mean) / std + bias

class AdaptiveBatchNorm(nn.Module):
  def __init__(self, in_size, ada_size):
    super(AdaptiveBatchNorm, self).__init__()
    self.scale = nn.Linear(ada_size, in_size)
    self.bias = nn.Linear(ada_size, in_size)

  def forward(self, inputs, style):
    in_view = inputs.view(inputs.size(0), -1)
    mean = in_view.mean(dim=0)
    std = in_view.mean(dim=0)
    scale = self.scale(style).view(style.size(0), -1, 1, 1)
    bias = self.bias(style).view(style.size(0), -1, 1, 1)
    return scale * (inputs - mean) / std + bias
