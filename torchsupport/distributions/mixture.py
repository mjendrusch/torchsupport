import torch
import torch.jit
from torch.distributions import constraints, Categorical, RelaxedOneHotCategorical
from torch.distributions.distribution import Distribution

class Mixture(Distribution):
  has_rsample = False
  def __init__(self, distributions, weights):
    self.distributions = distributions
    if all(map(lambda x: x.has_rsample, self.distributions)):
      self.has_rsample = True
    self.weights = weights

  def log_prob(self, value):
    result_exp = 0.0
    for weight, distribution in zip(self.weights.permute(1, 0), self.distributions):
      prob = distribution.log_prob(value).exp()
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

  def rsample(self):
    if not self.has_rsample:
      raise NotImplementedError("Mixture does not support rsample.")
    samples = []
    for distribution in self.distributions:
      sample = distribution.rsample().unsqueeze(0)
      samples.append(sample)
    samples = torch.cat(samples, dim=0)
    expand = samples.dim() - 2
    choice = RelaxedOneHotCategorical(probs=self.weights, temperature=0.1)
    choice = choice.rsample().permute(1, 0)
    choice = choice.view(choice.size(0), choice.size(1), *expand)
    result = (samples * choice).sum(dim=0)
    return result
