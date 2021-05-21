import torch
from torchsupport.flex.checkpointing.savable import (
  savable_of, Savable, SaveStateError
)

@savable_of(torch.nn.Module)
class SaveModule(Savable):
  def __init__(self, module):
    if isinstance(module, torch.nn.DataParallel):
      module = module.module
    self.module = module

  def write(self, data, name):
    for param in self.module.parameters():
      if torch.isnan(param).any():
        raise SaveStateError("Encountered NaN weights!")
    data[name] = self.module.state_dict()

  def read(self, data, name):
    self.module.load_state_dict(data[name])

@savable_of(torch.Tensor)
class SaveTensor(Savable):
  def __init__(self, tensor):
    self.tensor = tensor

  def write(self, data, name):
    data[name] = self.tensor

  def read(self, data, name):
    self.tensor[:] = data[name]
