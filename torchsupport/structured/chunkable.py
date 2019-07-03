import torch
from torch.nn.parallel.scatter_gather import Scatter

def chunk_sizes(lengths, num_targets):
  num_entities = len(lengths)
  chops = num_entities // num_targets
  result = [
    sum(lengths[idx * chops:(idx + 1) * chops])
    for idx in range(num_targets)
  ]
  return result

def chunk_tensor(tensor, lengths, targets, dim=0):
  return Scatter.apply(targets, lengths, dim, tensor)

class Chunkable():
  def chunk(self, targets):
    raise NotImplementedError("Abstract.")

def scatter_chunked(inputs, target_gpus, dim=0):
  r"""
  Slices tensors into approximately equal chunks and
  distributes them across given GPUs. Duplicates
  references to objects that are not tensors.
  """
  def scatter_map(obj):
    if isinstance(obj, Chunkable):
      return obj.chunk(target_gpus)
    if isinstance(obj, torch.Tensor):
      return Scatter.apply(target_gpus, None, dim, obj)

    if isinstance(obj, tuple) and len(obj) > 0:
      return list(zip(*map(scatter_map, obj)))
    if isinstance(obj, list) and len(obj) > 0:
      return list(map(list, zip(*map(scatter_map, obj))))
    if isinstance(obj, dict) and len(obj) > 0:
      return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
    return [obj for targets in target_gpus]

  try:
    return scatter_map(inputs)
  finally:
    scatter_map = None

def scatter_chunked_kwargs(inputs, kwargs, target_gpus, dim=0):
  r"""Scatter with support for kwargs dictionary"""
  inputs = scatter_chunked(inputs, target_gpus, dim) if inputs else []
  kwargs = scatter_chunked(kwargs, target_gpus, dim) if kwargs else []
  if len(inputs) < len(kwargs):
    inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
  elif len(kwargs) < len(inputs):
    kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
  inputs = tuple(inputs)
  kwargs = tuple(kwargs)
  return inputs, kwargs
