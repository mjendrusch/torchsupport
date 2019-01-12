import torch
import torch.nn as nn
import torch.nn.functional as func

class Router(nn.Module):
  def __init__(self, predictor, experts):
    """Mixture-of-experts hard-attention routing module.
    Distributes an input onto a set of "expert" modules,
    given a prediction, which expert is most relevant to a given input.

    Args:
      predictor (nn.Module): hard attention module.
      experts (iterable): iterable of expert modules.
    """
    super(Router, self).__init__()
    self.predictor = predictor
    self.experts = nn.ModuleList(experts)

  def forward(self, predicate, input):
    index = func.argmax(self.predictor(predicate))
    return self.experts[input](index)

class SoftRouter(nn.Module):
  def __init__(self, predictor, experts, top_n = 3):
    """Mixture-of-experts sparse soft-attention routing module.
    Distributes an input onto a set of "expert" modules,
    given a prediction, which expert is most relevant to
    a given input, averaging over the `top_n` most relevant experts.
    
    Args:
      predictor (nn.Module): soft attention module.
      experts (iterable): iterable of expert modules.
      top_n (int): number of experts for averaging.
    """
    super(SoftRouter, self).__init__()
    self.predictor = predictor
    self.experts = experts
    self.top_n = top_n
  
  def forward(self, predicate, input):
    indices = list(range(len(self.experts)))
    prediction = self.predictor(predicate)
    indices.sort(key=lambda x: prediction[x], reverse=True)
    tops = indices[0:self.top_n]
    H = prediction[tops]
    exps = torch.func.exp(-H)
    sumval = exps.sum()
    exps /= sumval
    result = exps[0] * self.experts[indices[0]](input)
    for idx in range(1, self.top_n):
      result += exps[idx] * self.experts[indices[idx]](input)
