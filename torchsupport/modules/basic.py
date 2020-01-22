"""Implements basic neural network building blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as func

class MLP(nn.Module):
  """Multilayer perceptron module."""
  def __init__(self, in_size, out_size,
               hidden_size=128, depth=3,
               activation=func.relu,
               normalization=lambda x: x,
               batch_norm=True):
    """Multilayer preceptron module.

    Args:
      in_size (int): number of input features.
      out_size (int): number of output features.
      hidden_size (int): number of hidden features.
      depth (int): number of perceptron layers.
      activation: activation function.
      batch_norm (bool): use batch normalization?
    """
    super(MLP, self).__init__()
    self.activation = activation
    self.bn = None

    if isinstance(hidden_size, list):
      self.blocks = nn.ModuleList([
        normalization(nn.Linear(in_size, hidden_size[0]))
      ] + [
        normalization(nn.Linear(hidden_size[idx], hidden_size[idx + 1]))
        for idx in range(len(hidden_size) - 1)
      ])
      self.postprocess = nn.Linear(hidden_size[-1], out_size)

      if batch_norm:
        self.bn = nn.ModuleList([
          nn.BatchNorm1d(hidden_size[idx])
          for idx in range(len(hidden_size))
        ])
    else:
      self.blocks = nn.ModuleList([
        normalization(nn.Linear(in_size, hidden_size))
      ] + [
        normalization(nn.Linear(hidden_size, hidden_size))
        for _ in range(depth - 2)
      ])
      self.postprocess = normalization(nn.Linear(hidden_size, out_size))

      if batch_norm:
        self.bn = nn.ModuleList([
          nn.BatchNorm1d(hidden_size)
        ] + [
          nn.BatchNorm1d(hidden_size)
          for _ in range(depth - 2)
        ])

  def forward(self, inputs):
    out = inputs
    if self.bn is not None:
      for bn, block in zip(self.bn, self.blocks):
        out = bn(self.activation(block(out)))
    else:
      for block in self.blocks:
        out = self.activation(block(out))
    return self.postprocess(out)

def one_hot_encode(data, code, numeric=False):
  """Encodes a sequence in one-hot format, given a code.

  Args:
    data (Sequence): sequence of non-encoded data points.
    code (Int | Sequence): sequence of codes for one-hot encoding or a the alphabet length if data is already a list of indices.
  Returns:
    FloatTensor containing a one-hot encoding of the input data.
  """
  try:
    if isinstance(code, int):
      coded = torch.tensor(data, dtype=torch.int64)
      alpha_len = code
    else:
      coded = torch.tensor(list(map(
        code.index, data
      )), dtype=torch.int64)
      alpha_len = len(code)
  except:
    print(data)
    exit()
  if numeric:
    result = coded
  else:
    data_len = len(data)
    result = torch.zeros(alpha_len, data_len, dtype=torch.float)
    result[coded, torch.arange(data_len)] = 1
  return result

class OneHotEncoder(nn.Module):
  """Basic sequence encoder into the one-hot format."""
  def __init__(self, code, numeric=False):
    """Basic sequence encoder into the one-hot format.

    Args:
      code (Sequence): sequence giving the one-hot encoding.
      numeric (bool): return LongTensor instead?
    """
    super(OneHotEncoder, self).__init__()
    self.code = code
    self.numeric = numeric

  def forward(self, sequence):
    return one_hot_encode(sequence, self.code,
                          numeric=self.numeric)
