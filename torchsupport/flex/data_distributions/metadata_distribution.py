import itertools

import torch
import torch.nn as nn
from torch.distributions import Distribution, Categorical
from torch.utils.data import Dataset, DataLoader

from torchsupport.data.io import to_device, DeviceMovable
from torchsupport.flex.data_distributions.data_distribution import InfiniteSampler

class MetaDataset(Dataset):
  def __init__(self, data):
    super().__init__()
    self.data = data

  def __getitem__(self, index):
    return super().__getitem__(index), index

  def __len__(self):
    return len(self.data)

class MetaDataDistribution(nn.Module, Distribution, DeviceMovable):
  @staticmethod
  def init_meta(meta):
    meta.zero_()

  def move_to(self, device):
    self.device = device
    return self

  def __init__(self, data_set, meta_type=None,
               batch_size=1, device="cpu", init_meta=None,
               **kwargs):
    super().__init__()
    self.device = device
    self.batch_size = batch_size
    init_meta = init_meta or MetaDataDistribution.init_meta
    self.data_set = MetaDataset(data_set)
    self.meta_type = meta_type or torch.Size(1)
    self.meta_data = nn.Parameter(torch.zeros(
      len(self.data_set), *meta_type,
      requires_grad=True
    ))
    with torch.no_grad():
      init_meta(self.meta_data)
    self.loader = DataLoader(
      data_set, batch_size=batch_size, drop_last=True,
      sampler=InfiniteSampler(data_set), **kwargs
    )
    self.iter = iter(self.loader)

  def sample(self, sample_shape=torch.Size()):
    data, indices = next(self.iter)
    meta_data = self.meta_data[indices]
    return to_device((data, meta_data, indices), self.device)

class WeightedInfiniteSampler:
  def __init__(self, data_set, weights):
    self.size = len(data_set)
    self.weights = weights

  def __iter__(self):
    yield from itertools.islice(self.permutation(), 0, None, 1)

  def permutation(self):
    while True:
      with torch.no_grad():
        dist = Categorical(logits=self.weights)
        sample = dist.sample(self.batch_size).view(-1)
      yield from sample

class WeightedDataDistribution(nn.Module, Distribution, DeviceMovable):
  @staticmethod
  def init_meta(meta):
    meta.normal_()

  def move_to(self, device):
    self.device = device
    return self

  def __init__(self, data_set, batch_size=1, device="cpu",
               init_meta=None, **kwargs):
    super().__init__()
    self.device = device
    init_meta = init_meta or WeightedDataDistribution.init_meta
    self.data_set = MetaDataset(data_set)
    self.weight = nn.Parameter(torch.zeros(
      len(self.data_set), requires_grad=True
    ))
    self.weight.share_memory_()
    with torch.no_grad():
      init_meta(self.weight)
    self.loader = DataLoader(
      data_set, batch_size=batch_size, drop_last=True,
      sampler=WeightedInfiniteSampler(data_set, self.weight), **kwargs
    )
    self.iter = iter(self.loader)

  def sample(self, sample_shape=torch.Size()):
    data, indices = next(self.iter)
    weight = self.weight.log_softmax(dim=0)
    weight = weight[indices]
    return to_device((data, weight, indices), self.device)
