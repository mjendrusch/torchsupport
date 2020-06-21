from copy import copy, deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.data.namedtuple import namedtuple

class Energy(nn.Module):
  data_type = namedtuple("Data", [
    "batch", "energy", "args"
  ])
  sample_type = namedtuple("SampleData", [
    "data", "args"
  ])
  def __init__(self, energy, keep_rate=0.95, device="cpu"):
    super().__init__()
    self.energy = energy
    self.device = device
    self.keep_rate = keep_rate

    self.example_batch, self.example_args = self._example_batch()
    self.example_energy = self._example_energy()

  def _example_batch(self):
    return self.prepare(1)

  def _example_energy(self):
    with torch.no_grad():
      pass_batch, pass_args = self.pack_batch(
        self.example_batch, self.example_args
      )
      return self.energy(pass_batch, *pass_args)

  def move(self):
    result = deepcopy(self)
    result.energy = self.energy.clone_to(self.device)
    return result

  def push(self):
    self.energy.push()

  def pull(self):
    self.energy.pull()

  def prepare(self, batch_size):
    data = torch.rand_like(batch_size, *self.shape)
    return self.sample_type(
      data=data, args=None
    )

  def batch_size(self, batch, args):
    return batch.size(0)

  def pack_batch(self, batch, args):
    args = args or []
    return batch, args

  def unpack_batch(self, batch):
    return batch

  def recombine_batch(self, batch, args, new_batch, new_args, drop):
    batch[drop] = new_batch
    if args:
      args[drop] = new_args
    return batch, args

  def reset(self, batch, args, energy):
    size = self.batch_size(batch, args)
    drop = self.keep_rate < torch.rand(size)
    drop_count = int(drop.sum())
    new_batch, new_args = self.prepare(drop_count)
    batch, args = self.recombine_batch(
      batch, args, new_batch, new_args, drop
    )
    return batch, args

  def schema(self):
    args = self.example_args or [None]
    return self.data_type(
      batch=self.example_batch[0],
      energy=self.example_energy[0],
      args=args[0]
    )

  def forward(self, data, *args):
    return self.energy(data, *args)
