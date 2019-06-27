# Adapted from PyTorch:

import torch
from torch.utils.data import DataLoader as TorchDataLoader

class Collatable():
  @classmethod
  def collate(cls, inputs):
    raise NotImplementedError("Abstract.")

  @classmethod
  def cat(cls, inputs):
    assert all(map(lambda x: x.__class__ is inputs[0].__class__, inputs))
    the_class = inputs[0].__class__
    return the_class.collate(inputs)

class BatchFirst(Collatable, torch.Tensor):
  @classmethod
  def collate(cls, inputs):
    return torch.cat([
      tensor
      for tensor in inputs
    ], dim=0)

error_message_format = (
  "default_collate: batch must contain tensors, structures, numpy arrays, "
  "numbers, dicts or lists; found {}"
)

def default_collate(batch):
  elem = batch[0]
  elem_type = type(elem)
  if isinstance(elem, dict):
    return {key: default_collate([d[key] for d in batch]) for key in elem}
  elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
    return elem_type(*(default_collate(samples) for samples in zip(*batch)))
  elif isinstance(elem, (list, tuple)):
    transposed = zip(*batch)
    return [default_collate(samples) for samples in transposed]
  elif isinstance(elem, Collatable):
    return Collatable.cat(batch)
  else:
    return torch.utils.data.dataloader.default_collate(batch)
  raise TypeError(error_message_format.format(elem_type))

def DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
               batch_sampler=None, num_workers=0, collate_fn=default_collate,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None):
  return TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                         sampler=sampler, batch_sampler=batch_sampler,
                         num_workers=num_workers, collate_fn=collate_fn,
                         pin_memory=pin_memory, drop_last=drop_last,
                         timeout=timeout, worker_init_fn=worker_init_fn)
