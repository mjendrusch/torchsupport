import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as func

class IntermediateExtractor(nn.Module):
  def __init__(self, module, submodules):
    """Extract results of intermediate submodules for a given module.

    Args:
      module (nn.Module): module for extraction.
      submodules (list string): list of submodule names for extraction.
    """
    super(IntermediateExtractor, self).__init__()
    self.module = module
    self.submodules = submodules
    if self.submodules == "all":
      self.submodules = []
      for name, child in self.module.named_children():
        self.submodules.append(name)

    self.outputs = []
    def hook(module, input, output):
      self.outputs.append((module._ts_tracking_name, output))
    for submodule in self.submodules:
      self.modules.__dict__[submodule]._ts_tracking_name = name
      self.modules.__dict__[submodule].register_forward_hook(hook)

  def forward(self, input):
    out = self.module(input)
    outputs = self.outputs + [("result", out)]
    self.outputs = []
    return outputs

class ResNextBlockNd(nn.Module):
  def __init__(self, in_size, out_size, hidden_size, N=1,
               cardinality=32, activation=func.elu, **kwargs):
    super(ResNextBlockNd, self).__init__()

    assert out_size >= in_size

    conv = getattr(nn, f"Conv{N}d")
    bn = getattr(nn, f"BatchNorm{N}d")
    self.blocks = nn.ModuleList([
      conv(in_size, hidden_size * cardinality, 1),
      conv(
        hidden_size * cardinality,
        hidden_size * cardinality,
        3, groups=cardinality,
        **kwargs
      ),
      conv(hidden_size * cardinality, out_size, 1)
    ])
    self.bn = nn.ModuleList([
      bn(in_size),
      bn(hidden_size * cardinality),
      bn(hidden_size * cardinality)
    ])
    self.activation = activation
    self.out_size = out_size
    self.in_size = in_size

  def forward(self, inputs):
    out = inputs
    for bn, block in zip(self.bn, self.blocks):
      out = block(self.activation(bn(out)))
    if self.out_size > self.in_size:
      filler = torch.zeros(
        inputs.size(0),
        self.out_size - self.in_size,
        *inputs.shape[2:]
      )
      inputs = torch.cat((inputs, filler), dim=1)
    return out + inputs

class ResNextBlock1d(ResNextBlockNd):
  def __init__(self, in_size, out_size, hidden_size,
               cardinality=32, activation=func.elu, **kwargs):
    super(ResNextBlock1d, self).__init__(
      in_size, out_size, hidden_size, cardinality=cardinality, N=1, activation=activation, **kwargs
    )

class ResNextBlock2d(ResNextBlockNd):
  def __init__(self, in_size, out_size, hidden_size,
               cardinality=32, activation=func.elu, **kwargs):
    super(ResNextBlock2d, self).__init__(
      in_size, out_size, hidden_size, cardinality=cardinality, N=2, activation=activation, **kwargs
    )

class ResNextBlock3d(ResNextBlockNd):
  def __init__(self, in_size, out_size, hidden_size,
               cardinality=32, activation=func.elu, **kwargs):
    super(ResNextBlock3d, self).__init__(
      in_size, out_size, hidden_size, cardinality=cardinality, N=3, activation=activation, **kwargs
    )

class ResNetBlockNd(ResNextBlockNd):
  def __init__(self, in_size, out_size, hidden_size, N=1,
               activation=func.elu, **kwargs):
    super(ResNetBlockNd, self).__init__(
      self, in_size, out_size, hidden_size, N=N, activation=activation, **kwargs
    )

class ResNetBlock1d(ResNetBlockNd):
  def __init__(self, in_size, out_size, hidden_size,
               cardinality=32, activation=func.elu, **kwargs):
    super(ResNetBlock1d, self).__init__(
      in_size, out_size, hidden_size, cardinality=cardinality, N=1, activation=activation, **kwargs
    )

class ResNetBlock2d(ResNetBlockNd):
  def __init__(self, in_size, out_size, hidden_size,
               cardinality=32, activation=func.elu, **kwargs):
    super(ResNetBlock2d, self).__init__(
      in_size, out_size, hidden_size, cardinality=cardinality, N=2, activation=activation, **kwargs
    )

class ResNetBlock3d(ResNetBlockNd):
  def __init__(self, in_size, out_size, hidden_size,
               cardinality=32, activation=func.elu, **kwargs):
    super(ResNetBlock3d, self).__init__(
      in_size, out_size, hidden_size, cardinality=cardinality, N=3, activation=activation, **kwargs
    )
