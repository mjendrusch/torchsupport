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

class FixUpBlockNd(nn.Module):
  def __init__(self, in_size, out_size, N=1, index=0,
               activation=func.relu, **kwargs):
    super(FixUpBlockNd, self).__init__()

    conv = getattr(nn, f"Conv{N}d")
    
    self.convs = nn.ModuleList([
      conv(in_size, out_size, 3, **kwargs),
      conv(out_size, out_size, 3, **kwargs),
    ])
    self.project = lambda x: x if in_size == out_size else conv(in_size, out_size, 1)

    self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
    self.biases = nn.ParameterList([
      nn.Parameter(torch.tensor(0.0, dtype=torch.float))
      for _ in range(4)
    ])

    with torch.no_grad():
      self.convs[0].weight.data = self.convs[0].weight.data * (index + 1) ** (-0.5)
      self.convs[1].weight.data.zero_()
    
    self.activation = activation

  def forward(self, inputs):
    out = inputs + self.biases[0]
    out = self.convs[0](out) + self.biases[1]
    out = self.activation(out) + self.biases[2]
    out = self.scale * self.convs[1](out) + self.biases[3]
    return out + self.project(inputs)

class FixUpBlock1d(FixUpBlockNd):
  def __init__(self, in_size, out_size, index=0,
               activation=func.relu, **kwargs):
    super().__init__(
      in_size, out_size, N=1, index=index, activation=activation, **kwargs
    )

class FixUpBlock2d(FixUpBlockNd):
  def __init__(self, in_size, out_size, index=0,
               activation=func.relu, **kwargs):
    super().__init__(
      in_size, out_size, N=2, index=index, activation=activation, **kwargs
    )

class FixUpBlock3d(FixUpBlockNd):
  def __init__(self, in_size, out_size, index=0,
               activation=func.relu, **kwargs):
    super().__init__(
      in_size, out_size, N=3, index=index, activation=activation, **kwargs
    )

class FixUpFactory:
  def __init__(self, N=1):
    self.fixup = getattr(sys.modules[__name__], f"FixUpBlock{N}d")
    self.index = 0

  def __call__(self, *args, **kwargs):
    result = self.fixup(*args, index=self.index, **kwargs)
    self.index += 1
    return result

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
      ).to(inputs.device)
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
      self, in_size, out_size, hidden_size, cardinality=1, N=N, activation=activation, **kwargs
    )

class ResNetBlock1d(ResNetBlockNd):
  def __init__(self, in_size, out_size, hidden_size,
               activation=func.elu, **kwargs):
    super(ResNetBlock1d, self).__init__(
      in_size, out_size, hidden_size, N=1, activation=activation, **kwargs
    )

class ResNetBlock2d(ResNetBlockNd):
  def __init__(self, in_size, out_size, hidden_size,
               cardinality=32, activation=func.elu, **kwargs):
    super(ResNetBlock2d, self).__init__(
      in_size, out_size, hidden_size, N=2, activation=activation, **kwargs
    )

class ResNetBlock3d(ResNetBlockNd):
  def __init__(self, in_size, out_size, hidden_size,
               cardinality=32, activation=func.elu, **kwargs):
    super(ResNetBlock3d, self).__init__(
      in_size, out_size, hidden_size, N=3, activation=activation, **kwargs
    )
