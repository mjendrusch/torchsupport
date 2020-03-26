import torch
import torch.jit
from torch.distributions import constraints, Categorical
from torch.distributions.distribution import Distribution

class Mixture(Distribution):
  has_rsample = False
  def __init__(self, distributions, weights):
    self.distributions = distributions
    self.weights = weights

  def log_prob(self, value):
    result_exp = 0.0
    for weight, distribution in zip(self.weights.permute(1, 0), self.distributions):
      prob = distribution.log_prob(value).exp()
      print(prob)
      result_exp += weight[:, None] * prob
    result = torch.log(result_exp + 1e-6)
    return result

  def sample(self):
    samples = []
    for distribution in self.distributions:
      sample = distribution.sample().unsqueeze(0)
      samples.append(sample)
    samples = torch.cat(samples, dim=0)
    choice = Categorical(probs=self.weights)
    choice = choice.sample()
    result = samples[choice, torch.arange(samples.size(1))]
    return result
