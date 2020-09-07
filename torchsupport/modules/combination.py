import torch
import torch.nn as nn
import torch.nn.functional as func
from torchsupport.ops.shape import batchexpand, flatten

class Combination(nn.Module):
  r"""Structural element for disjoint combination / evaluation networks.

  Args:
    combinator (nn.Module): module joining two disjoint input tensors,
      for example by concatenation.
    evaluator (nn.Module): module taking a combined tensor, and performing
      computation on that tensor.
  """
  def __init__(self, combinator, evaluator):
    super(Combination, self).__init__()
    self.combinator = combinator
    self.evaluator = evaluator

  def forward(self, input, task):
    combination = self.combinator(input, task)
    result = self.evaluator(combination)
    return result

class Concatenation(Combination):
  r"""Structural element concatenating two tensors.

  Args:
    evaluator (nn.Module): module taking the concatenated tensor, and
      performing computation on that tensor.
  """
  def __init__(self, evaluator):
    super(Concatenation, self).__init__(
      lambda input, task: _concatenate(input, task),
      evaluator
    )
  
  def forward(self, input, task):
    return super(Concatenation, self).forward(input, task)

def _concatenate(input, task):
  flattened_input = flatten(input, batch=True)
  flattened_task = flatten(task, batch=False)
  concatenated = torch.cat([
    flattened_input.unsqueeze(1),
    batchexpand(flattened_task, flattened_input).unsqueeze(1)
  ], 1)
  return concatenated

class ConnectedCombination(Concatenation):
  r"""Structural element performing a linear map on a concatenation
  of two Tensors.
  
  Args:
    evaluator (nn.Module): module taking a combined tensor, and
      performing computation on that tensor.
    inputs (int): number of input features.
    outputs (int): desired number of output features.
    batch_norm (bool): perform batch normalization?
  """
  def __init__(self, evaluator, inputs, outputs, batch_norm=True):
    super(ConnectedCombination, self).__init__(evaluator)
    self.connected = nn.Linear(inputs, outputs)
    if batch_norm:
      self.batch_norm = nn.BatchNorm1d(outputs)
    else:
      self.batch_norm = None

  def forward(self, input, task):
    concatenated = _concatenate(input, task)
    combined = self.connected(concatenated)
    combined = func.dropout(combined, training=True)
    if self.batch_norm != None:
      combined = self.batch_norm(flatten(combined, batch=True)).unsqueeze(1)
    result = self.evaluator(combined)
    return result

class BilinearCombination(Combination):
  r"""Structural element combining two tensors by bilinear transformation.

  Args:
    evaluator (nn.Module): module taking a combined tensor, and
                performing computation on that tensor.
    inputs (list or tuple): number of input features for each input tensor.
    outputs (int): number of output features.
    batch_norm (bool): perform batch normalization?
  """
  def __init__(self, evaluator, inputs, outputs, batch_norm=True):
    bilinear = nn.Bilinear(*inputs, outputs)
    if batch_norm:
      batch_norm_layer = nn.BatchNorm1d(outputs)
    else:
      batch_norm_layer = None
    super(BilinearCombination, self).__init__(
      lambda input, task: self.compute(input, task),
      evaluator
    )
    self.bilinear = bilinear
    self.batch_norm = batch_norm_layer

  def compute(self, input, task):
    flattened_input = flatten(input, batch=True)
    flattened_task = flatten(task, batch=False)
    flattened_task = batchexpand(flattened_task, flattened_input)
    result = self.bilinear(flattened_input, flattened_task)
    if self.batch_norm != None:
      result = self.batch_norm(result)
    return result

  def forward(self, input, task):
    return super(BilinearCombination, self).forward(input, task)
