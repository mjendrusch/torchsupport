import torch
from torch.distributions.distribution import Distribution
from torch.distributions.kl import register_kl

from torchsupport.data.match import Matchable, match

class DistributionList(Distribution):
  has_rsample = True
  def __init__(self, items):
    self.items = items

  def match(self, other):
    result = 0.0
    for s, o in zip(self.items, other.items):
      match_result = match(s, o)
      result = result + match_result
    return result

  def log_prob(self, value):
    log_prob = 0.0
    for dist, val in zip(self.items, value):
      current = dist.log_prob(val)
      current = current.view(current.size(0), -1).sum(dim=1)
      log_prob = log_prob + current
    return log_prob

  def sample(self, sample_shape=torch.Size()):
    return [
      dist.sample(sample_shape=sample_shape)
      for dist in self.items
    ]

  def rsample(self, sample_shape=torch.Size()):
    return [
      dist.rsample(sample_shape=sample_shape)
      for dist in self.items
    ]
