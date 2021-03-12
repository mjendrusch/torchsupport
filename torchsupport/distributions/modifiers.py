import torch
from torchsupport.modules.gradient import replace_gradient

def fixed(distribution, sample):
  def return_sample(dist, sample_shape=torch.Size()):
    return sample
  distribution.rsample = return_sample
  distribution.sample = return_sample
  return distribution

def hardened(distribution):
  def harden_sample(dist, sample_shape=torch.Size()):
    result = dist.rsample(sample_shape=sample_shape)
    return replace_gradient(dist.harden(result), result)
  distribution.rsample = harden_sample
  distribution.sample = harden_sample
