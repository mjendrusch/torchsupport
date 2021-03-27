import torch
from torch.distributions import RelaxedOneHotCategorical, Categorical, kl_divergence, register_kl

@register_kl(RelaxedOneHotCategorical, RelaxedOneHotCategorical)
def kl_relaxed_one_hot_categorical(p, q):
  p = Categorical(probs=p.probs)
  q = Categorical(probs=q.probs)
  return kl_divergence(p, q)
