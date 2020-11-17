import torch
import torch.nn.functional as func
from torch.distributions import kl_divergence, Distribution
from torchsupport.data.collate import Collatable
from torchsupport.data.io import DeviceMovable

class Matchable:
  def match(self, other):
    raise NotImplementedError("Abstract.")

  def rmatch(self, other):
    return self.match(other)

class MatchableList(Matchable):
  def __init__(self, items):
    self.items = items

  def match(self, other):
    result = 0.0
    if isinstance(other, MatchableList):
      for x, y in zip(self, other):
        result = result + match(x, y)
    else:
      for x in self:
        result = result + match(x, other)
    return result

  def __getitem__(self, index):
    return self.items[index]

  def __len__(self):
    return len(self.items)

def match_lp(x, y, p=2):
  result = x - y
  if result.dim() > 1:
    result = result.view(x.size(0), -1).norm(p=p, dim=1)
    result = result.mean(dim=0)
  else:
    result = result.norm(p=p, dim=0)
  return result

def match(p, q):
  if isinstance(p, Matchable):
    return p.match(q)
  if isinstance(q, Matchable):
    return q.match(p)
  if isinstance(p, Distribution):
    if isinstance(q, Distribution):
      try:
        return kl_divergence(p, q).mean(dim=0).sum()
      except NotImplementedError as e:
        sample = p.rsample()
        log_p = p.log_prob(sample)
        log_q = q.log_prob(sample)
        return (log_p - log_q).mean(dim=0).sum()
    else:
      return -p.log_prob(q).mean(dim=0).sum()
  if isinstance(q, Distribution):
    return -q.log_prob(p).mean(dim=0).sum()
  raise ValueError(f"Neither p nor q admit a matching operation.")

def match_l2(x, y):
  return match_lp(x, y, p=2)

def match_l1(x, y):
  return match_lp(x, y, p=1)

def match_bce(x, y):
  result = func.binary_cross_entropy_with_logits(y, x, reduction="none")
  if result.dim() > 1:
    result = result.view(x.size(0), -1).sum(dim=1)
    result = result.mean(dim=0)
  else:
    result = result.sum()
  return result

class MatchTensor(DeviceMovable, Collatable, Matchable):
  def __init__(self, tensor, match=match_l2):
    self.tensor = tensor
    self._match = match

  def move_to(self, target):
    return MatchTensor(self.tensor.to(target), match=self._match)

  @classmethod
  def collate(cls, inputs):
    this_match = inputs[0]._match
    result = torch.cat(tuple(map(lambda x: x.tensor.unsqueeze(0), inputs)), dim=0)
    result = cls(result, match=this_match)
    return result

  def __getattr__(self, name):
    if name == "tensor":
      self.tensor = torch.tensor(0.0)
    return getattr(self.tensor, name)

  def __add__(self, other):
    return torch.add(self, other)

  def __pow__(self, other):
    return self.tensor ** other

  def __sub__(self, other):
    return torch.sub(self, other)

  def __mul__(self, other):
    return torch.add(self, other)

  def __truediv__(self, other):
    return torch.div(self, other)

  def __radd__(self, other):
    return torch.add(other, self)

  def __rpow__(self, other):
    return other ** self.tensor

  def __rsub__(self, other):
    return torch.sub(other, self)

  def __rmul__(self, other):
    return torch.add(other, self)

  def __rtruediv__(self, other):
    return torch.div(other, self)

  def match(self, other):
    if isinstance(other, MatchTensor):
      result = self._match(self.tensor, other.tensor)
      result = result + other._match(other.tensor, self.tensor)
    else:
      result = self._match(self.tensor, other)
    return result

  def __torch_function__(self, func, types, args=(), kwargs=None):
    if kwargs is None:
      kwargs = {}
    args = [a.tensor if hasattr(a, 'tensor') else a for a in args]
    ret = func(*args, **kwargs)
    return ret
