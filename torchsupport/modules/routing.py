import torch
import torch.nn as nn
import torch.nn.functional as func

class Router(nn.Module):
  """
    Router(predictor, [experts...])
  Mixture-of-experts routing module, distributing an input onto
  a set of "expert" modules, given a prediction, which expert is
  most relevant to a given input.
  """
  def __init__(self, predictor, experts):
    super(self, Router).__init__(self)
    self.predictor = predictor
    self.experts = experts

  def forward(self, predicate, input):
    index = func.argmax(self.predictor(predicate))
    return self.experts[input](index)

class SoftRouter(nn.Module):
  """
    SoftRouter(predictor, [experts...], top_n = 3)
  Mixture-of-experts routing module, distributing an input onto
  a set of "expert" modules, given a prediction, which expert is
  most relevant to a given input, averaging over the `top_n` most
  relevant experts.
  """
  def __init__(self, predictor, experts, top_n = 3):
    super(self, SoftRouter).__init__(self)
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
