import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils.spectral_norm import spectral_norm

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
    expand = [1] * (inputs.dim() - 2)
    mean = inputs.mean(dim=1, keepdim=True)
    std = inputs.std(dim=1, keepdim=True)
    scale = self.scale(style).view(style.size(0), -1, *expand)
    scale = scale - scale.mean(dim=1, keepdim=True) + 1
    bias = self.bias(style).view(style.size(0), -1, *expand)
    bias = bias - bias.mean(dim=1, keepdim=True)
    return scale * (inputs - mean) / (std + 1e-6) + bias

class NotNorm(nn.Module):
  def __init__(self, in_size):
    super().__init__()
    self.in_size = in_size

  def forward(self, inputs):
    extension = [1] * (inputs.dim() - 2)

    out = inputs.view(inputs.size(0), inputs.size(1), -1)
    mean = out.mean(dim=-1, keepdim=True)
    std = out.std(dim=-1, keepdim=True)
    normed = (out - mean.detach()) / std.detach()
    out = std * normed + mean
    return out.view(inputs.shape)

class AdaNorm(nn.Module):
  def __init__(self, in_size, normalization=None):
    super().__init__()
    normalization = normalization or spectral_norm
    self.in_size = in_size
    self.scale = normalization(nn.Linear(2 * in_size, in_size))
    self.bias = normalization(nn.Linear(2 * in_size, in_size))

  def forward(self, inputs):
    out = inputs.view(inputs.size(0), inputs.size(1), -1)
    mean = out.mean(dim=-1)
    std = out.std(dim=-1)
    normed = (out - mean.unsqueeze(-1).detach()) / std.unsqueeze(-1).detach()

    features = torch.cat((mean, std), dim=1)
    mean = self.bias(features).unsqueeze(-1)
    std = self.scale(features).unsqueeze(-1)

    out = std * normed + mean
    return out.view(inputs.shape)

class AdaDataNorm(nn.Module):
  def __init__(self, in_size, normalization=None):
    super().__init__()
    normalization = normalization or spectral_norm
    self.in_size = in_size
    self.mean = nn.Parameter(torch.zeros(1, in_size, 1))
    self.std = nn.Parameter(torch.zeros(1, in_size, 1))
    self.scale = normalization(nn.Linear(2 * in_size, in_size))
    self.bias = normalization(nn.Linear(2 * in_size, in_size))

  def forward(self, inputs):
    out = inputs.view(inputs.size(0), inputs.size(1), -1)
    mean = out.mean(dim=-1)
    std = out.std(dim=-1)
    normed = (out - self.mean) / self.std

    features = torch.cat((mean, std), dim=1)
    mean = self.bias(features).unsqueeze(-1)
    std = self.scale(features).unsqueeze(-1)

    out = std * normed + mean
    return out.view(inputs.shape)

class ScaleNorm(nn.Module):
  def __init__(self, *args):
    super().__init__()
    self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float))

  def forward(self, inputs):
    out = inputs.view(inputs.size(0), -1)
    norm = out.norm(dim=1, keepdim=True)
    out = self.scale * out / (norm + 1e-16)
    return out.view(*inputs.shape)

class BotchNorm(nn.Module):
  def __init__(self, in_size, normalization=None):
    super().__init__()
    normalization = normalization or spectral_norm
    self.bn = nn.BatchNorm2d(in_size, affine=False)
    self.scale = normalization(nn.Linear(2 * in_size, in_size))
    self.bias = normalization(nn.Linear(2 * in_size, in_size))

  def forward(self, inputs):
    out = inputs.view(inputs.size(0), inputs.size(1), -1)
    mean = out.mean(dim=-1)
    std = out.std(dim=-1)

    out = self.bn(inputs)
    out = out.view(out.size(0), out.size(1), -1)

    features = torch.cat((mean, std), dim=1)
    mean = self.bias(features).unsqueeze(-1)
    std = self.scale(features).unsqueeze(-1)

    out = std * out + mean
    return out.view(inputs.shape)

class FilterResponseNorm(nn.Module):
  def __init__(self, in_size, eps=1e-16):
    super().__init__()
    self.eps = eps
    self.in_size = in_size
    self.register_parameter(
      "scale",
      nn.Parameter(torch.ones(in_size, dtype=torch.float))
    )
    self.register_parameter(
      "bias",
      nn.Parameter(torch.zeros(in_size, dtype=torch.float))
    )
    self.register_parameter(
      "threshold",
      nn.Parameter(torch.zeros(in_size, dtype=torch.float))
    )

  def forward(self, inputs):
    out = inputs.view(inputs.size(0), inputs.size(1), -1)
    nu2 = (out ** 2).mean(dim=-1)
    extension = [1] * (inputs.dim() - 2)
    denominator = torch.sqrt(nu2 + self.eps)
    denominator = denominator.view(inputs.size(0), inputs.size(1), *extension)
    scale = self.scale.view(1, self.scale.size(0), *extension)
    bias = self.bias.view(1, self.bias.size(0), *extension)
    threshold = self.threshold.view(1, self.threshold.size(0), *extension)
    out = inputs / denominator.detach()
    out = func.relu(scale * out + bias - threshold) + threshold
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
