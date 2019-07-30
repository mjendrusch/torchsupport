from copy import copy
import torch
from torchsupport.data.collate import Collatable
from torchsupport.data.io import DeviceMovable
from torchsupport.structured.chunkable import (
  Chunkable, chunk_sizes, chunk_tensor
)

class PackedTensor(DeviceMovable, Collatable, Chunkable):
  def __init__(self, tensors, lengths=None, split=True):
    self.tensor = tensors
    self.split = split
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
    return PackedTensor(data, lengths=lengths)

  def move_to(self, device):
    the_copy = copy(self)
    the_copy.tensor = self.tensor.to(device)
    return self

  def chunk(self, targets):
    print("CHUNKING PACKED", self.split, self.lengths, targets)
    sizes = chunk_sizes(self.lengths, len(targets))
    print("CHUNKING PACKED", sizes)
    chunks = chunk_tensor(self.tensor, sizes, targets, dim=0)
    result = []
    offset = 0
    step = len(self.lengths) // len(targets)
    for chunk in chunks:
      the_tensor = PackedTensor(chunk)
      the_tensor.lengths = self.lengths[offset:offset + step]
      result.append(the_tensor.tensor)
    print("CHUNKING PACKED", len(result))
    return result
