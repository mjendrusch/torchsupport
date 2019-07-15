import torch
import torch.nn as nn
import torch.nn.functional as func

def _compute_indices(data, indices, counts, max_count):
  offset = max_count - counts
  offset = offset.roll(1, 0)
  offset[0] = 0
  offset = torch.repeat_interleave(offset.cumsum(dim=0), counts, dim=0)

  index = offset + torch.arange(len(indices))
  return index

def pad(data, indices, value=0):
  unique, counts = indices.unique(return_counts=True)
  result_indices = unique
  max_count = counts.max()
  index = _compute_indices(data, indices, counts, max_count)
  result = torch.zeros(
    len(counts), max_count, *data.shape[1:],
    dtype=data.dtype, device=data.device
  )
  result.view(-1, *data.shape[1:])[index] = data
  return result, result_indices, counts

def pack(data, indices):
  unique, counts = indices.unique(return_counts=True)
  result_indices = unique
  tensors = []
  offset = 0
  for count in counts:
    tensors.append(data[offset:offset + count])
    offset += count
  result = nn.utils.rnn.pack_sequence(
    tensors, enforce_sorted=False
  )
  return result, result_indices, counts

def repack(data, indices, target_indices):
  out = torch.zeros(
    target_indices.size(0), *data.shape[1:],
    dtype=data.dtype, device=data.device
  )
  unique, lengths = indices.unique(return_counts=True)
  unique, target_lengths = target_indices.unique(return_counts=True)
  offset = target_lengths - lengths
  offset = offset.roll(1, 0)
  offset[0] = 0
  offset = torch.repeat_interleave(offset.cumsum(dim=0), lengths, dim=0)
  index = offset + torch.arange(len(indices))

  out[index] = data
  return data, target_indices

HAS_SCATTER = True
try:
  import torch_scatter as tsc
  def _scatter_op(operation):
    def _op(*args, **kwargs):
      return operation(*args, **kwargs, dim=0)
    try:
      return _op
    finally:
      _op = None
  add = _scatter_op(tsc.scatter_add)
  sub = _scatter_op(tsc.scatter_sub)
  mul = _scatter_op(tsc.scatter_mul)
  min = _scatter_op(tsc.scatter_min)
  max = _scatter_op(tsc.scatter_max)
  div = _scatter_op(tsc.scatter_div)
  std = _scatter_op(tsc.scatter_std)
  mean = _scatter_op(tsc.scatter_mean)
except ImportError:
  def _scatter_op(operation, update, value=0):
    def _op(data, indices, dim=0, out=None, dim_size=None, fill_value=value):
      padded, pad_indices, counts = pad(data, indices, value=value)
      processed = operation(padded, counts.unsqueeze(1))
      if dim_size is None:
        dim_size = processed.size(0)
      if out is None:
        out = torch.zeros(
          dim_size, *processed.shape[1:],
          dtype=data.dtype,
          device=data.device
        )
        out.fill_(fill_value)
      out = update(out, processed, pad_indices)
      return out
    try:
      return _op
    finally:
      _op = None

  def _update_add(out, processed, indices):
    out[indices] += processed
    return out

  def _update_mul(out, processed, indices):
    out[indices] *= processed
    return out

  def _update_min(out, processed, indices):
    out[indices] = torch.min(out[indices], processed)
    return out

  def _update_max(out, processed, indices):
    out[indices] = torch.max(out[indices], processed)
    return out

  def _update_over(out, processed, indices):
    out[indices] = processed
    return out

  add = _scatter_op(lambda x, y: x.sum(dim=1), _update_add, value=0)
  sub = _scatter_op(lambda x, y: -x.sum(dim=1), _update_add, value=0)
  mean = _scatter_op(lambda x, y: x.sum(dim=1) / y.float(), _update_add, value=0)
  min = _scatter_op(lambda x, y: x.min(dim=1).values, _update_min, value=float("inf"))
  max = _scatter_op(lambda x, y: x.max(dim=1).values, _update_max, value=float("-inf"))
  mul = _scatter_op(lambda x, y: x.prod(dim=1), _update_mul, value=1)
  div = _scatter_op(lambda x, y: 1.0 / x.prod(dim=1), _update_mul, value=1)

  def var(data, indices, dim=0, out=None, dim_size=None, unbiased=True):
    if out is None and dim_size is None:
      out = torch.zeros(
        indices.max() + 1, *data.shape[1:],
        dtype=data.dtype, device=data.device
      )
    if dim_size is None:
      dim_size = out.size(0)
    _, counts = indices.unique(return_counts=True)
    counts = counts.float()
    factor = counts / (counts - 1)
    factor = factor.unsqueeze(1)
    mean_value = mean(data, indices, dim_size=dim_size)[indices]
    out = factor * mean((data - mean_value) ** 2, indices, dim_size=dim_size)
    return out

  def std(data, indices, dim=0, out=None, dim_size=None, unbiased=True):
    return torch.sqrt(var(
      data, indices, dim=dim, out=out, dim_size=dim_size, unbiased=unbiased
    ))

def softmax(data, indices, dim_size=None):
  if dim_size is None:
    dim_size = indices.max() + 1
  out = data - max(data, indices, dim_size=dim_size)[indices]
  out = out.exp()
  out = out / (add(out, indices, dim_size=dim_size)[indices] + 1e-16)
  return out

def sequential(module, data, indices):
  packed, _, _ = pack(data, indices)
  result, hidden = module(packed)
  return result.data, hidden.data

def reduced_sequential(module, data, indices, out=None, dim_size=None):
  packed, pack_indices, counts = pack(data, indices)
  result, hidden = module(packed)
  last = torch.cumsum(counts, dim=0) - 1

  if dim_size is None:
    dim_size = indices.max() + 1
  if out is None:
    out = torch.zeros(
      dim_size, result.shape[1:],
      dtype=data.dtype, device=data.device
    )
  out_hidden = torch.zeros_like(out)
  out[pack_indices] += result.data[last]
  out_hidden[pack_indices] += hidden.data[0]

  return out, out_hidden

def batched(module, data, indices, padding_value=0):
  padded, _, counts = pad(data, indices, value=padding_value)
  result = module(padded.transpose(1, 2)).transpose(1, 2)
  packed = nn.utils.rnn.pack_padded_sequence(result, counts, batch_first=True)
  result = packed.data
  return result

def reduced_batched(module, data, indices, out=None,
                    dim_size=None, padding_value=0):
  padded, pad_indices, counts = pad(data, indices, value=padding_value)
  result = module(padded.transpose(1, 2)).transpose(1, 2)
  result = result.sum(dim=1) / counts.float()
  if dim_size is None:
    dim_size = result.size(0)
  if out is None:
    out = torch.zeros(
      dim_size, *result.shape[1:],
      dtype=data.dtype, device=data.device
    )
  out[pad_indices] += result
  return out

class ScatterModule(nn.Module):
  def __init__(self):
    """Applies a reduction function to the neighbourhood of each entity."""
    super(ScatterModule, self).__init__()

  def reduce(self, source):
    raise NotImplementedError("Abstract")

  def forward(self, source, indices):
    results = torch.zeros_like(source)
    for idx in range(results.size(0)):
      results += self.reduce(source[(indices == idx).nonzero()])
    return results
