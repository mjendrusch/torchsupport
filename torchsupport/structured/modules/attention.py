import math

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.rezero import ReZero

def cross_attention(query, key, value, target_mask=None, source_mask=None, mask=None):
  dot = torch.einsum("ihjk,ihjl->ihkl", query, key) / math.sqrt(key.size(2))

  # mask softmax:
  if (target_mask is not None) or (source_mask is not None) or (mask is not None):
    if source_mask is None:
      source_mask = torch.ones(1, 1, key.size(-1), device=key.device)
    if target_mask is None:
      target_mask = torch.ones(1, 1, query.size(-1), device=query.device)
    available_mask = torch.einsum("ijk,ijl->ijkl", target_mask, source_mask)
    if mask is None:
      mask = available_mask
    else:
      mask = mask * available_mask
    dot = mask * dot - (1 - mask) * 1e6
    amap = dot.softmax(dim=-1) * mask
  amap = dot.softmax(dim=-1)

  return torch.einsum("ihkl,ihvl->ihvk", amap, value)

class RawCrossAttention(nn.Module):
  def __init__(self, size=128, heads=8):
    super().__init__()
    self.heads = heads
    self.size = size
    self.query = nn.Conv1d(size, size, 1)
    self.key = nn.Conv1d(size, size, 1)
    self.value = nn.Conv1d(size, size, 1)

  def forward(self, target, source, target_mask=None, source_mask=None, mask=None):
    query = self.query(target)
    key = self.key(source)
    value = self.value(source)
    query = query.view(query.size(0), self.heads, -1, query.size(2))
    key = key.view(key.size(0), self.heads, -1, key.size(2))
    value = value.view(value.size(0), self.heads, -1, value.size(2))
    result = cross_attention(
      query, key, value,
      target_mask=target_mask,
      source_mask=source_mask,
      mask=mask
    )
    return result.reshape(result.size(0), -1, result.size(-1))

class CrossAttention(RawCrossAttention):
  def __init__(self, size=128, out_size=None, heads=8):
    super().__init__(size=size, heads=heads)
    out_size = out_size or size
    self.output = nn.Conv1d(size, out_size, 1, bias=False)

  def forward(self, target, source, target_mask=None, source_mask=None, mask=None):
    out = super().forward(
      target, source,
      target_mask=target_mask,
      source_mask=source_mask,
      mask=mask
    )
    out = self.output(out)
    if target_mask is not None:
      out = out * target_mask
    return out

class SelfAttention(CrossAttention):
  def forward(self, target, target_mask=None, mask=None):
    return super().forward(
      target, target,
      target_mask=target_mask,
      source_mask=target_mask,
      mask=mask
    )

class ReZeroCrossAttention(nn.Module):
  def __init__(self, size=128, heads=8):
    super().__init__()
    self.attention = CrossAttention(
      size=size, out_size=size, heads=heads
    )
    self.zero = ReZero(size)

  def forward(self, target, source, target_mask=None, source_mask=None, mask=None):
    update = self.attention(
      target, source,
      target_mask=target_mask,
      source_mask=source_mask,
      mask=mask
    )
    return self.zero(target, update)

class ReZeroSelfAttention(ReZeroCrossAttention):
  def forward(self, target, target_mask=None, mask=None):
    return super().forward(
      target, target,
      target_mask=target_mask,
      source_mask=target_mask,
      mask=mask
    )

class SimplexAttention(nn.Module):
  def __init__(self, size=128, heads=8):
    super().__init__()
    self.cross_attention = CrossAttention(
      size=size, out_size=2 * size, heads=heads
    )
    self.zero = ReZero(2 * size)

  def forward(self, target, source, target_mask=None, source_mask=None, mask=None):
    attention = self.cross_attention(
      target, source,
      target_mask=target_mask,
      source_mask=source_mask,
      mask=mask
    )
    attention = self.zero(torch.zeros_like(attention), attention)
    scale, bias = attention.chunk(2, dim=1)
    return (1 + scale) * target + bias

class DuplexAttention(nn.Module):
  def __init__(self, size=128, heads=8):
    super().__init__()
    self.heads = heads
    self.cross_attention = CrossAttention(
      size=size, out_size=2 * size, heads=heads
    )
    self.output = nn.Conv1d(size, 2 * size, 1, bias=False)
    self.zero = ReZero(2 * size)

  def forward(self, target, source, target_mask=None, source_mask=None, mask=None):
    forward_attention = self.cross_attention(
      source, target,
      target_mask=source_mask,
      source_mask=target_mask,
      mask=mask
    )
    key, value = forward_attention.chunk(2, dim=1)
    # introduce dummy head dimension
    key = key[:, None]
    value = value[:, None]
    query = target[:, None]

    attention = cross_attention(
      query, key, value,
      target_mask=target_mask,
      source_mask=source_mask,
      mask=mask
    )[:, 0]
    attention = self.output(attention)
    if target_mask is not None:
      attention = attention * target_mask
    attention = self.zero(torch.zeros_like(attention), attention)
    scale, bias = attention.chunk(2, dim=1)
    return (1 + scale) * target + bias
