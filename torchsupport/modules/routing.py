import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.gradient import replace_gradient

class Router(nn.Module):
  r"""Conditional execution of a set of expert modules.
    Distributes an input onto a set of "expert" modules
    given a prediction which expert is most relevant to
    a given input.

    Shape:
      - Inputs: :math:`(N, S^{in}_{experts}...)`
      - Condition: :math:`(N, S_{predictor}...)`
      - Outputs: :math:`(N, S^{out}_{experts}...)`

    Args:
      predictor (nn.Module): hard attention module.
      experts (iterable): iterable of expert modules.

    Examples:
      >>> experts = [nn.Linear(128, 10) for idx in range(5)]
      >>> predictor = nn.Linear(128, 5)
      >>> router = Router(predictor, experts)
      >>> data = torch.randn(32, 128)
      >>> condition = torch.randn(32, 128)
      >>> result = router(data, condition)

    .. note::
        gradients through the expert selection are estimated
        using a simple straight-through gradient estimator.
  """
  def __init__(self, predictor, experts):
    super(Router, self).__init__()
    self.predictor = predictor
    self.experts = nn.ModuleList(experts)

  def _combine_experts(self, inputs, prediction):
    r"""Computes results for all experts with straight-through
    gradient estimation."""

    multiply = torch.ones_like(prediction)
    multiply = replace_gradient(multiply, prediction)
    multiply = multiply.permute(1, 0).unsqueeze(-1)

    combined = torch.cat([
      multiply * expert(inputs).unsqueeze(0)
      for expert in self.experts
    ], dim=0)
    return combined

  def forward(self, inputs, condition):
    prediction = self.predictor(condition)
    index = prediction.argmax()
    ind = torch.arange(prediction.size(0), device=condition.device)
    combined = self._combine_experts(inputs, prediction)
    result = combined[index, ind]
    return result

class SoftRouter(nn.Module):
  r"""Soft conditional execution of a set of expert modules.
    Distributes an input onto a set of "expert" modules
    given a prediction which expert is most relevant to
    a given input.

    Shape:
      - Inputs: :math:`(N, S^{in}_{experts}...)`
      - Condition: :math:`(N, S_{predictor}...)`
      - Outputs: :math:`(N, S^{out}_{experts}...)`

    Args:
      predictor (nn.Module): hard attention module.
      experts (iterable): iterable of expert modules.
      top_n (int): number of top experts to combine.
        Default: 3

    Examples:
      >>> experts = [nn.Linear(128, 10) for idx in range(5)]
      >>> predictor = nn.Linear(128, 5)
      >>> router = Router(predictor, experts)
      >>> data = torch.randn(32, 128)
      >>> condition = torch.randn(32, 128)
      >>> result = router(data, condition)

    .. note::
        gradients through the expert selection are computed
        only for the top k outputs.
  """
  def __init__(self, predictor, experts, top_n=3):
    super(SoftRouter, self).__init__()
    self.predictor = predictor
    self.experts = experts
    self.top_n = top_n

  def forward(self, inputs, condition):
    indices = list(range(len(self.experts)))
    prediction = self.predictor(condition)
    indices.sort(key=lambda x: prediction[x], reverse=True)
    tops = indices[0:self.top_n]
    H = prediction[tops]
    exps = torch.func.exp(-H)
    sumval = exps.sum()
    exps /= sumval
    result = exps[0] * self.experts[indices[0]](inputs)
    for idx in range(1, self.top_n):
      result += exps[idx] * self.experts[indices[idx]](inputs)
