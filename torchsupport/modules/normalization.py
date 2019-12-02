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
    return scale * (inputs - mean) / (std + 1e-6) + bias

class AdaptiveInstanceNormPP(AdaptiveInstanceNorm):
  def __init__(self, in_size, ada_size):
    super(AdaptiveInstanceNormPP, self).__init__(in_size, ada_size)
    self.mean_scale = nn.Linear(ada_size, in_size)

  def forward(self, inputs, style):
    in_view = inputs.view(inputs.size(0), inputs.size(1), 1, 1, -1)
    mean = in_view.mean(dim=-1)
    mean_mean = mean.mean(dim=1, keepdim=True)
    std = in_view.std(dim=-1)
    mean_std = mean.std(dim=1, keepdim=True)
    scale = self.scale(style).view(style.size(0), -1, 1, 1)
    mean_scale = self.mean_scale(style).view(style.size(0), -1, 1, 1)
    bias = self.bias(style).view(style.size(0), -1, 1, 1)
    result = scale * (inputs - mean) / (std + 1e-6) + bias
    correction = mean_scale * (mean - mean_mean) / (mean_std + 1e-6)
    return result + correction

class AdaptiveBatchNorm(nn.Module):
  def __init__(self, in_size, ada_size):
    super(AdaptiveBatchNorm, self).__init__()
    self.scale = nn.Linear(ada_size, in_size)
    self.bias = nn.Linear(ada_size, in_size)

  def forward(self, inputs, style):
    in_view = inputs.view(inputs.size(0), -1)
    mean = inputs.mean(dim=0, keepdim=True)
    std = inputs.std(dim=0, keepdim=True)
    scale = self.scale(style).view(style.size(0), -1, 1, 1)
    scale = scale - scale.mean(dim=1, keepdim=True) + 1
    bias = self.bias(style).view(style.size(0), -1, 1, 1)
    bias = bias - bias.mean(dim=1, keepdim=True)
    return scale * (inputs - mean) / (std + 1e-6) + bias

class AdaptiveLayerNorm(nn.Module):
  def __init__(self, in_size, ada_size):
    super(AdaptiveLayerNorm, self).__init__()
    self.scale = nn.Linear(ada_size, in_size)
    self.bias = nn.Linear(ada_size, in_size)
  
  def forward(self, inputs, style):
    mean = inputs.mean(dim=1, keepdim=True)
    std = inputs.std(dim=1, keepdim=True)
    scale = self.scale(style).view(style.size(0), -1, 1, 1)
    scale = scale - scale.mean(dim=1, keepdim=True) + 1
    bias = self.bias(style).view(style.size(0), -1, 1, 1)
    bias = bias - bias.mean(dim=1, keepdim=True)
    return scale * (inputs - mean) / (std + 1e-6) + bias

class FilterResponseNorm(nn.Module):
  def __init__(self, in_size, eps=1e-16):
    super().__init__()
    self.eps = eps
    self.in_size = in_size
    self.register_parameter(
      "scale",
      nn.Parameter(torch.tensor(1.0, dtype=torch.float))
    )
    self.register_parameter(
      "bias",
      nn.Parameter(torch.tensor(0.0, dtype=torch.float))
    )
    self.register_parameter(
      "threshold",
      nn.Parameter(torch.tensor(0.0, dtype=torch.float))
    )

  def forward(self, inputs):
    out = inputs.view(inputs.size(0), inputs.size(1), -1)
    nu2 = out.mean(dim=-1)
    extension = [1] * (inputs.dim() - 2)
    denominator = torch.sqrt(nu2 + self.eps)
    denominator = denominator.view(inputs.size(0), inputs.size(1), *extension)
    out = inputs / denominator
    out = func.relu(self.scale * out + self.bias - self.threshold) + self.threshold
    return out

class AdaptiveFilterResponseNorm(nn.Module):
  def __init__(self, in_size, ada_size, eps=1e-16):
    super().__init__()
    self.eps = eps
    self.in_size = in_size
    self.scale = nn.Linear(ada_size, in_size)
    self.bias = nn.Linear(ada_size, in_size)
    self.threshold = nn.Linear(ada_size, in_size)

  def forward(self, inputs, condition):
    out = inputs.view(inputs.size(0), inputs.size(1), -1)
    nu2 = out.mean(dim=-1)
    extension = [1] * (inputs.dim() - 2)
    denominator = torch.sqrt(nu2 + self.eps)
    denominator = denominator.view(inputs.size(0), inputs.size(1), *extension)
    out = inputs / denominator
    scale = self.scale(condition)
    bias = self.bias(condition)
    threshold = self.threshold(condition)
    out = func.relu(scale * out + bias - threshold) + threshold
    return out
