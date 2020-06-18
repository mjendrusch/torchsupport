from collections import namedtuple
from copy import copy, deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as func

class ModuleEnergy(nn.Module):
  data_type = namedtuple("Data", [
    "batch", "energy"
  ])
  def __init__(self, energy, shape=None, device="cpu"):
    super().__init__()
    self.energy = energy
    self.device = device
    self.shape = shape

    self.example_batch = self._example_batch()
    self.example_energy = self._example_energy()

  def _example_batch(self):
    return self.prepare(1)[0]

  def _example_energy(self):
    with torch.no_grad():
      return self.energy(self.prepare(1))[0]

  def move(self):
    result = deepcopy(self)
    result.energy = self.energy.clone_to(self.device)
    return result

  def push(self):
    self.energy.push()

  def pull(self):
    self.energy.pull()

  def prepare(self, batch_size):
    return torch.rand_like(batch_size, *self.shape)

  def schema(self):
    return self.energy.schema(
      self.example_batch, self.example_energy
    )
