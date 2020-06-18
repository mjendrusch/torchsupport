import os
from copy import deepcopy, copy

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from torchsupport.data.io import to_device
from torchsupport.interacting.control import ReadWriteControl

class InertModule(nn.Module):
  def __init__(self, module):
    super().__init__()
    self.module = module

  def schema(self):
    return self.module.schema()

  def forward(self, *args, **kwargs):
    return self.module(*args, **kwargs)

def _share_skip_inert(module):
  if isinstance(module, InertModule):
    pass
  else:
    module.share_memory()

class SharedModule(nn.Module):
  def __init__(self, module, dynamic=False):
    super().__init__()
    self.ctrl = ReadWriteControl(self)
    self.dynamic = dynamic
    self.source_process = os.getpid()
    self.shared_module = deepcopy(module).cpu().share_memory()
    self._module = InertModule(module)

  def __deepcopy__(self, memo):
    cls = self.__class__
    result = cls(deepcopy(self._module.module, memo), dynamic=self.dynamic)
    result.ctrl = self.ctrl.clone(result)
    result.source_process = self.source_process
    result.shared_module = self.shared_module
    return result

  def clone_to(self, target="cpu"):
    result = deepcopy(self)
    result._module = result._module.to(target)

    return result

  def is_clone(self):
    return os.getpid() != self.source_process

  def share_memory(self):
    self.apply(_share_skip_inert)

  def schema(self):
    return self._module.schema()

  def pull_changes(self):
    if self.ctrl.changed:
      shared_state_dict = self.shared_module.state_dict()
      self._module.module.load_state_dict(shared_state_dict)
      self.ctrl.advance()

  def push_changes(self):
    state_dict = self._module.module.state_dict()
    state_dict = to_device(state_dict, "cpu")
    self.shared_module.load_state_dict(state_dict)
    self.ctrl.change()

  def pull(self):
    with self.ctrl.read:
      pass # NOTE: just pull changes

  def push(self):
    with self.ctrl.write:
      pass # NOTE: just push changes

  def forward(self, *args, **kwargs):
    if self.dynamic:
      with self.ctrl.read:
        return self._module(*args, **kwargs)
    else:
      return self._module(*args, **kwargs)

# class SharedPolicy:
#   def __call__(self, state, inputs=None):

