import torch
import torch.nn as nn
import torch.nn.functional as func

def _compute_indices(data, indices, counts, max_count):
  offset = max_count - counts
  offset = offset.roll(1, 0)
  offset[0] = 0
  offset = torch.repeat_interleave(offset.cumsum(dim=0), counts, dim=0)
  offset = offset.to(data.device)

  index = offset + torch.arange(len(indices)).to(data.device)
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
  return result, result_indices, index, counts

def unpad(data, index):
  return data.contiguous().view(-1, *data.shape[2:])[index]

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
  index = offset + torch.arange(len(indices)).to(data.device)

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
      padded, pad_indices, _, counts = pad(data, indices, value=value)
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
  out = out / (add(out, indices, dim_size=dim_size)[indices] + 1e-6)
  return out

def autoregressive(module, data, indices):
  _e, counts = indices.unique(return_counts=True)
  max_count = counts.max()
  out = data
  values = []
  for idx in range(max_count):
    out = module(out)
    values.append(out.unsqueeze(0))

  values = torch.cat(values, dim=0)
  access_0 = torch.cat([
    torch.arange(0, count, dtype=torch.long, device=values.device)
    for count in counts
  ], dim=0)
  access_1 = indices
  result = values[access_0, access_1]
  return result

def sequential(module, data, indices):
  packed, _, _ = pack(data, indices)
  result, hidden = module(packed)
  return result.data, hidden

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
  padded, _, index, counts = pad(data, indices, value=padding_value)
  result = module(padded.transpose(1, 2)).transpose(1, 2)
  return unpad(result, index)

def reduced_batched(module, data, indices, out=None,
                    dim_size=None, padding_value=0):
  padded, pad_indices, _, counts = pad(data, indices, value=padding_value)
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

def pairwise(op, data, indices, padding_value=0):
  padded, _, _, counts = pad(data, indices, value=padding_value)
  padded = padded.transpose(1, 2)
  reference = padded.unsqueeze(-1)
  padded = padded.unsqueeze(-2)
  op_result = op(padded, reference)

  # batch indices into pairwise tensor:
  batch_indices = torch.arange(counts.size(0))
  batch_indices = torch.repeat_interleave(batch_indices, counts ** 2)

  # first dimension indices:
  first_offset = counts.roll(1)
  first_offset[0] = 0
  first_offset = torch.cumsum(first_offset, dim=0)
  first_offset = torch.repeat_interleave(first_offset, counts)
  first_indices = torch.arange(counts.sum()) - first_offset
  first_indices = torch.repeat_interleave(
    first_indices,
    torch.repeat_interleave(counts, counts)
  )

  # second dimension indices:
  second_offset = torch.repeat_interleave(counts, counts).roll(1)
  second_offset[0] = 0
  second_offset = torch.cumsum(second_offset, dim=0)
  second_offset = torch.repeat_interleave(second_offset, torch.repeat_interleave(counts, counts))
  second_indices = torch.arange((counts ** 2).sum()) - second_offset

  # extract tensor from padded result using indices:
  result = op_result[batch_indices, first_indices, second_indices]

  # access: cumsum(counts ** 2)[idx] + counts[idx] * idy + idz
  access_batch = (counts ** 2).roll(1)
  access_batch[0] = 0
  access_batch = torch.cumsum(access_batch, dim=0)
  access_first = counts

  access = (access_batch, access_first)

  return result, batch_indices, first_indices, second_indices, access

def pairwise_no_pad(op, data, indices):
  unique, counts = indices.unique(return_counts=True)
  expansion = torch.cumsum(counts, dim=0)
  expansion = torch.repeat_interleave(expansion, counts)
  offset = torch.arange(0, counts.sum(), device=data.device)
  expansion = expansion - offset - 1
  expanded = torch.repeat_interleave(data, expansion.to(data.device), dim=0)

  expansion_offset = counts.roll(1)
  expansion_offset[0] = 0
  expansion_offset = torch.repeat_interleave(expansion_offset, counts)
  expansion_offset = torch.repeat_interleave(expansion_offset, expansion)
  off_start = torch.repeat_interleave(torch.repeat_interleave(counts, counts) - expansion, expansion)
  access = torch.arange(expansion.sum(), device=data.device)
  access = access - torch.repeat_interleave(expansion.roll(1).cumsum(dim=0), expansion) + off_start + expansion_offset

  result = op(expanded, data[access.to(data.device)])
  return result, torch.repeat_interleave(indices, expansion, dim=0)

def pairwise_get(data, access, idx):
  index = access[0][idx[0]] + access[1][idx[0]] * idx[1] + idx[2]
  return data[index]
