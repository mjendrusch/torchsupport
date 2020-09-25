from functools import partial

import torch
from torch.distributions import kl_divergence, Distribution

class Matchable:
  def match(self, other):
    raise NotImplementedError("Abstract.")

  def rmatch(self, other):
    return other.match(self)

def match_lp(x, y, p=2):
  result = x - y
  result = result.view(x.size(0), -1).norm(p=p, dim=1)
  return result.mean(dim=0)

def match(p, q):
  if isinstance(p, Distribution):
    return kl_divergence(p, q)
  if isinstance(p, Matchable):
    return p.match(q)
  if isinstance(q, Matchable):
    return q.rmatch(p)
  raise ValueError(f"Neither p nor q admit a matching operation.")

def match_l2(x, y) -> "L2_matching_loss":
  return match_lp(x, y, p=2)

def match_l1(x, y) -> "L1_matching_loss":
  return match_lp(x, y, p=1)

def match_any(p, q) -> "matching_loss":
  return match(p, q)
