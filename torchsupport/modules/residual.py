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
