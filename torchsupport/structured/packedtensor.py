from copy import copy
import torch
from torchsupport.data.collate import Collatable
from torchsupport.data.io import DeviceMovable
from torchsupport.data.tensor_provider import TensorProvider
from torchsupport.structured.chunkable import (
  Chunkable, chunk_sizes, chunk_tensor
)

class PackedTensor(DeviceMovable, Collatable, Chunkable, TensorProvider):
  def __init__(self, tensors, lengths=None, split=True, box=False):
    self.tensor = tensors
    self.split = split
    self.box = box
    self.lengths = [len(tensors)]
    if isinstance(self.tensor, (list, tuple)):
      self.lengths = list(map(lambda x: x.size(0), tensors))
      self.tensor = torch.cat(self.tensor, dim=0)
    if lengths is not None:
      self.lengths = lengths

  @classmethod
  def collate(cls, tensors):
    data = [
      tensor.tensor
      for tensor in tensors
    ]
    lengths = [
      length
      for tensor in tensors
      for length in tensor.lengths
    ]
    if not tensors[0].split:
      return torch.cat(data, dim=0)
    return PackedTensor(data, lengths=lengths, box=tensors[0].box)

  def move_to(self, device):
    the_copy = copy(self)
    the_copy.tensor = self.tensor.to(device)
    return self

  def tensors(self):
    return [self.tensor]

  def chunk(self, targets):
    sizes = chunk_sizes(self.lengths, len(targets))
    chunks = chunk_tensor(self.tensor, sizes, targets, dim=0)
    result = []
    step = len(self.lengths) // len(targets)
    for idx, chunk in enumerate(chunks):
      the_tensor = PackedTensor(chunk, split=self.split, box=self.box)
      the_tensor.lengths = self.lengths[idx * step:(idx + 1) * step]
      the_tensor = the_tensor if self.box else the_tensor.tensor
      result.append(the_tensor)
    return result
