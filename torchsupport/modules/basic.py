r"""Implements some basic building blocks for data preprocessing
and feed-forward neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as func

class MLP(nn.Module):
  r"""Multilayer preceptron module.

  Applies a series of matrix multiplications interleaved
  with nonlinear activation and optional normalization
  to a batch of input tensors.

  Shape:
    - Input: :math:`(N, C_{in})`
    - Output: :math:`(N, C_{out})`

  Args:
    in_size (int): number of input features.
    out_size (int): number of output features.
    hidden_size (int or list): number of hidden features. Default: 128
    depth (int): number of perceptron layers. Default: 3
    activation: activation function. Default: torch.nn.functional.relu
    batch_norm (bool): use batch normalization? Default: True
  """
  def __init__(self, in_size, out_size,
               hidden_size=128, depth=3,
               activation=func.relu,
               normalization=lambda x: x,
               batch_norm=True):
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
  r"""Encodes a sequence in one-hot format, given a code.

  Args:
    data (Sequence): sequence of non-encoded data points.
    code (int | Sequence): sequence of codes for one-hot
      encoding or a the alphabet length if data is already a list of indices.
    numeric (bool): map sequence to tensor of indices instead?
      Default: False

  Returns:
    :class:`torch.Tensor` of type :class:`torch.float32`
      containing a one-hot encoding of the input data if numeric
      is true, otherwise :class:`torch.Tensor` of type
      :class:`torch.long` containing the indices of the one-hot
      encoding.
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
  r"""Encodes a sequence in one-hot format, given a code.

  Refer to :func:`torchsupport.modules.basic.one_hot_encode` for details.

  Args:
    code (int | Sequence): sequence of codes for one-hot
      encoding or a the alphabet length if data is already a list of indices.
    numeric (bool): map sequence to tensor of indices instead?
      Default: False
  """
  def __init__(self, code, numeric=False):
    super(OneHotEncoder, self).__init__()
    self.code = code
    self.numeric = numeric

  def forward(self, sequence):
    return one_hot_encode(sequence, self.code,
                          numeric=self.numeric)
