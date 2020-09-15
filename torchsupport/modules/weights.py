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
    normalization = torch.tensor(self.weight[0].numel(), dtype=inputs.dtype)
    self.module.weight = self.weight * self.lr_scale * torch.sqrt(self.gain / normalization)
    self.module.bias = self.bias * self.lr_scale
    return self.module(inputs)

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
