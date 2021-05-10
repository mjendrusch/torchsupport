import torch
import torch.distributions as dist
from torchsupport.distributions.mixture import Mixture
from torchsupport.distributions.von_mises import VonMises
from torchsupport.distributions.standard import StandardNormal
from torchsupport.distributions.structured import DistributionList
from torchsupport.distributions.vae_distribution import VAEDistribution
# from torchsupport.distributions.modifiers import fixed, hardened
from torchsupport.distributions.kl_divergence import kl_relaxed_one_hot_categorical

def _harden_one_hot(self, inputs):
  hard = torch.zeros_like(inputs)
  hard_index = inputs.argmax(dim=-1)
  hard[torch.arange(0, hard.size(0)), hard_index] = 1.0
  return hard.detach()

def _harden_bernoulli(self, inputs):
  logits = torch.log(inputs / (1 - inputs + 1e-16) + 1e-16)
  return (logits > 0).float().detach()

def _hard_categorical(self, dist):
  return dist.OneHotCategorical(logits=dist.logits)

def _hard_bernoulli(self, dist):
  return dist.Bernoulli(logits=dist.logits)

def _condtitional_categorical(self, hard):
  noise = -torch.log(torch.rand_like(self.logits) + 1e-16)
  on_condition = noise * hard
  off_condition = noise * (1 - hard)
  offset = on_condition.view(-1, hard.size(-1)).sum(dim=-1).view(*hard.shape[:-1], 1)
  off_condition = off_condition / (self.probs + 1e-16) - offset
  soft_conditional = -torch.log(on_condition + off_condition + 1e-16)
  return soft_conditional

def _conditional_bernoulli(self, hard):
  noise = torch.rand_like(hard)
  on_condition = noise * hard
  off_condition = noise * (1 - hard)
  on_condition = on_condition * self.probs + (1 - self.probs)
  off_condition = off_condition * (1 - self.probs)
  total = on_condition + off_condition
  soft_conditional = torch.log(self.probs / (1 - self.probs + 1e-16) + 1e-16)
  soft_conditional += torch.log(total / (1 - total + 1e-16) + 1e-16)
  return soft_conditional

setattr(dist.RelaxedOneHotCategorical, "harden", _harden_one_hot)
setattr(dist.RelaxedOneHotCategorical, "hard_distribution", _hard_categorical)
setattr(dist.RelaxedOneHotCategorical, "conditional_rsample", _condtitional_categorical)

setattr(dist.RelaxedBernoulli, "harden", _harden_bernoulli)
setattr(dist.RelaxedBernoulli, "hard_distribution", _hard_bernoulli)
setattr(dist.RelaxedBernoulli, "conditional_rsample", _conditional_bernoulli)
