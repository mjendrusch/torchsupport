import torch
from torchsupport.data.collate import Collatable
from torchsupport.structured.chunkable import (
  Chunkable, chunk_sizes, chunk_tensor
)

class PackedTensor(Collatable, Chunkable):
  def __init__(self, tensors, lengths=None):
    self.tensor = tensors
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
    return PackedTensor(data, lengths=lengths)

  def chunk(self, targets):
    sizes = chunk_sizes(self.lengths, len(targets))
    chunks = chunk_tensor(self, sizes, targets, dim=0)
    result = []
    offset = 0
    step = len(self.lengths) // len(targets)
    for chunk in chunks:
      the_tensor = PackedTensor(chunk)
      the_tensor.lengths = self.lengths[offset:offset + step]
      result.append(the_tensor)
    return result
