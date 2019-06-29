import torch
from torchsupport.data.collate import Collatable

class PackedTensor(torch.Tensor, Collatable):
  def __new__(cls, tensor, dim=0, requires_grad=False):
    if isinstance(tensor, (list, tuple)):
      tensor = torch.cat(tensor, dim)
    result = torch.Tensor._make_subclass(cls, tensor, requires_grad)
    return result

  def __init__(self, tensor, dim=0, **kwargs):
    if isinstance(tensor, (list, tuple)):
      length = torch.tensor([
        subitem
        for item in tensor
        for subitem in (
          item.index
          if isinstance(item, PackedTensor)
          else [item.size(dim)]
        )
      ], dtype=torch.long)
    elif isinstance(tensor, PackedTensor):
      length = tensor.index
    else:
      length = torch.tensor([tensor.size(dim)], dtype=torch.long)
    self.index = length
    self.pack_dim = dim

  @classmethod
  def collate(cls, instances):
    return PackedTensor(instances, dim=instances[0].pack_dim)

  def __deepcopy__(self, memo):
    if id(self) in memo:
      return memo[id(self)]
    else:
      result = type(self)(self.data.clone(), self.requires_grad)
      memo[id(self)] = result
      return result

  def __repr__(self):
    prefix = f"PackedTensor of lengths:\n{self.index}\ncontaining:\n"
    return prefix + super(PackedTensor, self).__repr__()

  def to(self, args):
    result = PackedTensor(super().to(args))
    result.index = self.index
    result.pack_dim = self.pack_dim
    return result

def packed(tensor, dim=0):
  return PackedTensor(tensor, dim=0)
