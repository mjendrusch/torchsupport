import itertools

import torch
from torch.distributions import Distribution
from torchsupport.data.collate import DataLoader
from torchsupport.data.io import DeviceMovable, to_device

class InfiniteSampler:
  def __init__(self, data_set):
    self.size = len(data_set)

  def __iter__(self):
    yield from itertools.islice(self.permutation(), 0, None, 1)

  def permutation(self):
    while True:
      yield from torch.randperm(self.size)

class DataDistribution(Distribution, DeviceMovable):
  r"""Data distribution based on the PyTorch DataLoader
  combined with a standard dataset. Allows for loading
  multiple batches in parallel.
  """
  def __init__(self, data_set, batch_size=1, device="cpu", **kwargs):
    self.data = data_set
    self.device = device
    self.loader = DataLoader(
      data_set, batch_size=batch_size, drop_last=True,
      sampler=InfiniteSampler(data_set), **kwargs
    )
    self.iter = iter(self.loader)

  def move_to(self, device):
    self.device = device
    return self

  def sample(self, sample_shape=torch.Size()):
    return to_device(next(self.iter), self.device)
