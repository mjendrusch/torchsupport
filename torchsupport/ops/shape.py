import torch

def flatten(input, batch=False):
  if batch:
    return input.view(input.size()[0], -1)
  else:
    return input.view(-1)

def batchexpand(input, batch):
  result = input.unsqueeze(0).expand(
    batch.size()[0],
    *input.size()
  )
  return result

def deshape(inputs):
  dimension = inputs.dim()
  drop = dimension - 2
  if drop == 0:
    return inputs, None
  permutation = [0] + [2 + idx for idx in range(drop)] + [1]
  permuted = inputs.permute(*permutation)
  shape = permuted.shape
  return permuted.reshape(-1, inputs.shape[-1]), shape

def reshape(inputs, shape):
  if shape is None:
    return inputs
  inputs = inputs.reshape(*shape[:-1], inputs.size(-1))
  dimension = inputs.dim()
  drop = dimension - 2
  permutation = [0, -1] + list(range(1, drop + 1))
  inputs = inputs.permute(*permutation).contiguous()
  return inputs
