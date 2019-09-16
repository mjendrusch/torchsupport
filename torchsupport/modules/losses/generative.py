import torch
from torch import nn as nn
from torch.nn import functional as func

def normalized_distance(data, distance):
  data = data.view(data.size(0), -1)
  reference = data[:, None, :]
  comparison = data[:, :, None]
  result = distance(reference, comparison)
  result = result / result.sum(dim=1, keepdim=True).detach()
  return result

class NormalizedDistance(nn.Module):
  def __init__(self, distance=None):
    super().__init__()
    self.distance = distance
    if self.distance is None:
      self.distance = lambda x, y: (x - y).norm(dim=-1)

  def forward(self, data):
    return normalized_distance(data, self.distance)

def normalized_diversity_loss(x, y, d_x, d_y, alpha=1.0):
  size = x.size(0)
  D_x = normalized_distance(x, d_x)
  D_y = normalized_distance(y, d_y)
  result = max(alpha * D_x - D_y, 0.0)
  result[torch.arange(0, size), torch.arange(0, size)] = 0.0
  result = result.sum() / (size * (size - 1))
  return result

class NormalizedDiversityLoss(nn.Module):
  def __init__(self, d_x, d_y, alpha=1.0):
    super().__init__()
    self.d_x = d_x
    self.d_y = d_y
    self.alpha = alpha

  def forward(self, x, y):
    return normalized_diversity_loss(x, y, self.d_x, self.d_y, self.alpha)
