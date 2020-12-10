import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.dynamic import (
  dynamic_linear, dynamic_conv1d, dynamic_conv2d, dynamic_conv3d
)

def initialize(module, **kwargs):
  """Performs per-parameter initialization on a given module.
  Module parameters can be addressed by their name in the keyword arguments,
    paired with a function to perform their initialization.

  Args:
    module (nn.Module): module to be initialised.
    kwargs (dict): keyword arguments consisting of the name of a module
      attribute, together with a function to be applied in place to that
      attribute.
  """
  for name, function in kwargs.items():
    weight = getattr(module, name)
    with torch.no_grad():
      function(weight)

class LREqualization(nn.Module):
  """Equalizes the learning rate of a given module at run time
    as applied in the ProGAN and StyleGAN architectures.

  Args:
    modules (nn.Module): module to regularise by learning rate normalization.
    gain (float): Gain as used in He normal initialization.
    lr_scale (float): factor by which to multiply the module's learning rate.

  Returns:
    A wrapper module performing the forward pass of the input module with
    learning rate equalization and scaling.
  """
  def __init__(self, module, gain=2.0, lr_scale=1.0):
    super().__init__()
    self.lr_scale = lr_scale
    self.module = module
    self.gain = gain
    with torch.no_grad():
      self.module.weight.normal_().mul_(1 / lr_scale)
      if self.module.bias is not None:
        self.module.bias.zero_()
    self.func, self.kwargs = get_module_kind(module)

  def forward(self, inputs):
    weight, bias = lr_equal_weight(
      self.module.weight, self.module.bias,
      gain=self.gain, lr_scale=self.lr_scale
    )
    return self.func(inputs, weight, bias, **self.kwargs)

def lr_equal_weight(weight, bias=None, gain=1.0, lr_scale=1.0):
  normalization = torch.tensor(weight[0].numel(), dtype=weight.dtype)
  weight = weight * lr_scale * gain * torch.sqrt(1 / normalization)
  # weight = nn.Parameter(weight)
  if bias is not None:
    bias = bias * lr_scale
    # bias = nn.Parameter(bias)
    return weight, bias
  return weight, None

def lr_equal(module, lr_scale=1.0):
  """Applies learning rate equalization to a given module, if it has a weight attribute.

  Args:
    modules (nn.Module): module to regularise by learning rate normalization.
    lr_scale (float): factor by which to multiply the module's learning rate.

  Returns:
    A wrapper module performing the forward pass of the input module with
    learning rate equalization and scaling.
  """
  if hasattr(module, "weight"):
    return LREqualization(module, lr_scale=lr_scale)
  return module

def get_module_kind(module):
  conv_names = ["stride", "dilation", "padding"]
  function = ...
  kwargs = {}
  if isinstance(module, nn.Linear):
    function = dynamic_linear
  elif isinstance(module, nn.Conv1d):
    function = dynamic_conv1d
    for name in conv_names:
      kwargs[name] = getattr(module, name)
  elif isinstance(module, nn.Conv2d):
    function = dynamic_conv2d
    for name in conv_names:
      kwargs[name] = getattr(module, name)
  elif isinstance(module, nn.Conv3d):
    function = dynamic_conv3d
    for name in conv_names:
      kwargs[name] = getattr(module, name)
  else:
    raise NotImplementedError(
      "Weight demodulation is only implemented for Linear and Conv layers."
    )
  return function, kwargs

class WeightDemodulation(LREqualization):
  def __init__(self, module, gain=2.0, lr_scale=1.0, epsilon=1e-6):
    super().__init__(module, gain=gain, lr_scale=lr_scale)
    self.epsilon = epsilon

  def forward(self, inputs, condition):
    weight, bias = lr_equal_weight(
      self.module.weight, self.module.bias,
      gain=self.gain, lr_scale=self.lr_scale
    )
    weight = demodulated_weight(weight, condition, epsilon=self.epsilon)
    bias = bias[None].expand(weight.size(0), bias.size(0)).contiguous()
    return self.func(inputs, weight, bias, **self.kwargs)

def demodulated_weight(weight, condition, epsilon=1e-6):
  shape = weight.shape
  weight = weight[None].view(1, *shape[:2], -1) * condition[:, None, :, None]
  sigma = (weight ** 2).view(*weight.shape[:2], -1).sum(dim=-1, keepdim=True)
  sigma = (sigma + epsilon).sqrt()
  sigma = sigma.view(*sigma.shape, 1)
  weight = weight / sigma
  weight = weight.view(condition.size(0), *shape)
  return weight
