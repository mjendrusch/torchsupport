import torch
import torch.nn as nn
import torch.nn.functional as func

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
    self.weight = module.weight
    self.bias = module.bias
    with torch.no_grad():
      self.weight.normal_().mul_(1 / lr_scale)
      self.bias.zero_()

  def forward(self, inputs):
    self.module.weight, self.module.bias = lr_equal_weight(
      self.weight, self.bias, gain=self.gain, lr_scale=self.lr_scale
    )
    return self.module(inputs)

def lr_equal_weight(weight, bias=None, gain=2.0, lr_scale=1.0):
  normalization = torch.tensor(weight[0].numel(), dtype=weight.dtype)
  weight = weight * lr_scale * torch.sqrt(gain / normalization)
  if bias:
    bias = bias * lr_scale
    return weight, bias
  return weight

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

class WeightDemodulation(LREqualization):
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
  def __init__(self, module, gain=2.0, lr_scale=1.0, epsilon=1e-6):
    super().__init__(module, gain=gain, lr_scale=lr_scale)
    self.epsilon = epsilon

  def forward(self, inputs, condition):
    weight, bias = lr_equal_weight(
      self.weight, self.bias, gain=self.gain, lr_scale=self.lr_scale
    )
    weight = demodulized_weight(weight, condition, epsilon=self.epsilon)
    self.module.weight = weight
    self.module.bias = bias
    return self.module(inputs)

def demodulized_weight(weight, condition, epsilon=1e-6):
  weight = weight[None] * condition[:, None, :, None, None]
  sigma = (weight ** 2).view(*weight.shape[:2], -1).sum(dim=-1)
  sigma = (sigma + epsilon).sqrt()
  sigma = sigma.view(*sigma.shape, 1, 1, 1)
  weight = weight / sigma
  return weight
