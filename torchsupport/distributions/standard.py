import torch
from torch.distributions import Distribution, Normal, RelaxedBernoulli, RelaxedOneHotCategorical

def StandardNormal(size):
  return Normal(
    torch.zeros(size),
    torch.ones(size)
  )
