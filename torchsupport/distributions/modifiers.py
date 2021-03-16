import types
import torch
from torchsupport.modules.gradient import replace_gradient

def fixed(distribution, sample):
  def return_sample(dist, sample_shape=torch.Size()):
    return sample
  distribution.rsample = types.MethodType(return_sample, distribution)
  distribution.sample = types.MethodType(return_sample, distribution)
  return distribution

def hardened(distribution):
  def harden_sample(dist, sample_shape=torch.Size()):
    result = dist.rsample(sample_shape=sample_shape)
    return replace_gradient(dist.harden(result), result)
  distribution.rsample = types.MethodType(harden_sample, distribution)
  distribution.sample = types.MethodType(harden_sample, distribution)
