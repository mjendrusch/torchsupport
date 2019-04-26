"""Implements basic neural network building blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as func

class MLP(nn.Module):
  """Multilayer perceptron module."""
  def __init__(self, in_size, out_size,
               hidden_size=128, depth=3,
               activation=func.relu,
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
    self.blocks = nn.ModuleList([
      nn.Linear(in_size, hidden_size)
    ] + [
      nn.Linear(hidden_size, hidden_size)
      for _ in range(depth - 2)
    ])
    self.postprocess = nn.Linear(hidden_size, out_size)
    self.activation = activation
    self.bn = None

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