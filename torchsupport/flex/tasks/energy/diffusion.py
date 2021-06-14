import torch
from torch.distributions import Normal

class Diffusion:
  def mixing(self, data, noise, level):
    return data

  def conditional(self, data, noise, level):
    return 0.0

class VEDiffusion(Diffusion):
  def mixing(self, data, noise, level):
    expand = [1] * (data.dim() - level.dim())
    level = level.view(*level.shape, *expand)
    return data + level * noise

  def conditional(self, data, condition, level):
    numerator = -(data - condition) ** 2
    denominator = 2 * level ** 2
    result = numerator / denominator
    return result.view(result.size(0), -1).sum(dim=1, keepdim=True)

class VPDiffusion(Diffusion):
  def mixing(self, data, noise, level):
    expand = [1] * (data.dim() - level.dim())
    level = level.view(*level.shape, *expand)
    return (1 - level).sqrt() * data + level.sqrt() * noise

  def conditional(self, data, condition, level):
    numerator = -((1 - level).sqrt() * data - condition) ** 2
    denominator = 2 * level
    result = numerator / denominator
    return result.view(result.size(0), -1).sum(dim=1, keepdim=True)

class DiscreteDiffusion(Diffusion):
  def mixing(self, data, noise, level):
    mask = (torch.rand_like(data) < level).float()
    return mask * noise + (1 - mask) * data

  def conditional(self, data, condition, level):
    dist = (1 - level) * condition + level * torch.ones_like(condition)
    result = (dist * data).sum(dim=1).log()
    return result.view(result.size(0), -1).sum(dim=1, keepdim=True)

class RandomReplacementDiffusion(DiscreteDiffusion):
  def __init__(self, base, sigma=1e-3):
    self.base = base
    self.sigma = sigma

  def conditional(self, data, condition, level):
    base = self.base.log_prob(data).exp()
    delta = Normal(condition, self.sigma)
    delta = delta.log_prob(data)
    delta = delta.view(delta.size(0), -1).sum(dim=1, keepdim=True).exp()
    prob = (1 - level) * delta + level * base
    return prob.log()

class ComposedDiffusion(Diffusion):
  def __init__(self, *components):
    self.components = components

  def mixing(self, data, noise, level):
    result = []
    for d, n, l, c in zip(data, noise, level, self.components):
      result.append(c(d, n, l))
    return result

  def conditional(self, data, condition, level):
    result = 0.0
    for d, n, l, c in zip(data, condition, level, self.components):
      result = result + c(d, n, l)
    return result
