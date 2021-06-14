from numpy.lib.utils import source
import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.rezero import ReZero
from torchsupport.structured.modules.attention import CrossAttention, ReZeroCrossAttention
from torchsupport.modules.unet import NestedModule, NestedWrapper, nest_sequential

class BasicResample(NestedModule):
  def __init__(self, size, dim=2, hole=None):
    super().__init__(hole)
    self.downsample = getattr(func, f"avg_pool{dim}d")
    self.upsample = func.interpolate

  def enter(self, inputs, *args, **kwargs):
    return self.downsample(inputs, 2), None

  def exit(self, inputs, skip, *args, **kwargs):
    return self.upsample(inputs, scale_factor=2, mode="bilinear")

def zero_module(x):
  with torch.no_grad():
    for param in x.parameters():
      param.zero_()
  return x

class ConditionalResBlock(nn.Module):
  def __init__(self, in_size, out_size, cond_size=512,
               dropout=0.1, dim=2, norm=True):
    super().__init__()
    conv = getattr(nn, f"Conv{dim}d")
    self.norm = None
    if norm:
      self.norm = nn.ModuleList([
        nn.GroupNorm(32, in_size),
        nn.GroupNorm(32, out_size)
      ])
    self.dropout = nn.Dropout(dropout)
    self.project_in = conv(in_size, out_size, 3, padding=1)
    self.condition = nn.Linear(cond_size, 2 * out_size)
    self.project_out = zero_module(conv(out_size, out_size, 3, padding=1))
    self.project = None
    if out_size != in_size:
      self.project = conv(in_size, out_size, 1, bias=False)

  def forward(self, inputs, time, *args, **kwargs):
    out = inputs
    if self.norm is not None:
      out = self.norm[0](out)
    out = self.project_in(func.silu(out))
    expand = (out.dim() - 2) * [1]
    cond = self.condition(time)
    cond = cond.view(*cond.shape, *expand)
    scale, bias = cond.chunk(2, dim=1)
    if self.norm is not None:
      out = self.norm[1](out)
    out = scale * out + bias
    out = self.dropout(func.silu(out))
    out = self.project_out(out)
    if self.project is not None:
      inputs = self.project(inputs)
    return inputs + out

def basic_res_block(in_size, out_size, **kwargs):
  return NestedWrapper(
    None,
    left=ConditionalResBlock(in_size, out_size, **kwargs),
    right=ConditionalResBlock(2 * out_size, in_size, **kwargs)
  )

class AttentionBlock(nn.Module):
  def __init__(self, size=512, heads=8, positional_embedding=False, norm=False):
    super().__init__()
    self.attn = CrossAttention(size=size, heads=heads)
    self.attn.output = zero_module(self.attn.output)
    self.norm = None
    if norm:
      self.norm = nn.GroupNorm(32, size)
    self.positional_embedding = None
    if positional_embedding:
      self.positional_embedding = ... # TODO

  def forward(self, inputs, *args, target_mask=None, source_mask=None, mask=None):
    shape = inputs.shape
    out = inputs
    if self.positional_embedding is not None:
      emb = self.positional_embedding(out)
      out = out + emb
    out = out.view(*shape[:2], -1)
    if self.norm is not None:
      out = self.norm(out)
    out = self.attn(
      out, out, target_mask=target_mask,
      source_mask=source_mask, mask=mask
    )
    out = out.view(*shape)
    return inputs + out

class AttentionBox(NestedModule):
  def __init__(self, size, heads=8, hole=None):
    super().__init__(hole=hole)
    self.fwd = AttentionBlock(size, heads=heads)
    self.rev = AttentionBlock(size, heads=heads)

  def enter(self, inputs, *args, **kwargs):
    return self.fwd(inputs), None

  def exit(self, inputs, skip, *args, **kwargs):
    return self.rev(inputs)

def basic_attn_block(size, **kwargs):
  return AttentionBox(size, **kwargs)

class MiddleBlock(nn.Module):
  def __init__(self, size, cond_size=512, heads=8, norm=False, dim=2):
    super().__init__()
    self.res = nn.ModuleList([
      ConditionalResBlock(size, size, cond_size=cond_size, norm=norm, dim=dim),
      ConditionalResBlock(size, size, cond_size=cond_size, norm=norm, dim=dim)
    ])
    self.attn = AttentionBlock(size=size, heads=heads, norm=norm)

  def forward(self, inputs, time, *args, **kwargs):
    out = self.res[0](inputs, time)
    out = self.attn(out)
    out = self.res[1](out, time)
    return out

class SinusoidalEmbedding(nn.Module):
  def __init__(self, cond_size=512, depth=2):
    super().__init__()
    self.blocks = nn.ModuleList([
      nn.Linear(1, cond_size)
    ] + [
      nn.Linear(cond_size, cond_size)
      for idx in range(depth - 1)
    ])

  def forward(self, inputs):
    out = inputs[:, None]
    for idx, block in enumerate(self.blocks):
      out = block(out).sin()
    return out

class ResolutionBlock(nn.Module):
  def __init__(self, base=32, subfactors=None, cond_size=512,
               dropout=0.1, dim=2, attention=False, heads=8,
               norm=False):
    super().__init__()
    blocks = []
    for in_factor, out_factor in zip(subfactors[:-1], subfactors[1:]):
      blocks.append(ConditionalResBlock(
        base * in_factor, base * out_factor,
        cond_size=cond_size, dropout=dropout,
        dim=dim, norm=norm
      ))
      if attention:
        blocks.append(AttentionBlock(
          size=base * out_factor, heads=heads, norm=norm
        ))
    self.blocks = nn.ModuleList(blocks)

  def forward(self, inputs, time, *args):
    out = inputs
    for block in self.blocks:
      out = block(out, time)
    return out

def resolution_block(base=32, subfactors=None, **kwargs):
  left = ResolutionBlock(base=base, subfactors=subfactors, **kwargs)
  subfactors = list(reversed(subfactors))
  subfactors[0] = 2 * subfactors[0]
  right = ResolutionBlock(base=base, subfactors=subfactors, **kwargs)
  return NestedWrapper(None, left=left, right=right)

class DiffusionUNetBackbone2(nn.Module):
  def __init__(self, base=32, factors=None, attention_levels=None,
               embedding_block=SinusoidalEmbedding,
               basic_block=resolution_block,
               resample_block=BasicResample,
               middle_block=MiddleBlock,
               dim=2, heads=8, cond_size=512, dropout=0.1, norm=False):
    super().__init__()
    self.embedding = embedding_block(cond_size)
    self.blocks = nest_sequential(*([
      item
      for idx, subfactors in enumerate(factors)
      for item in [
        basic_block(
          base=base, subfactors=subfactors, dim=dim,
          cond_size=cond_size, heads=heads, dropout=dropout,
          attention=idx in attention_levels, norm=norm
        )
      ] + [
        resample_block(base * subfactors[-1], dim=dim)
      ]
    ] + [
      middle_block(
        base * factors[-1][-1], cond_size=cond_size,
        dim=dim, norm=norm
      )
    ]))

  def forward(self, inputs, time):
    time = self.embedding(time)
    return self.blocks(inputs, time)

class DiffusionUNetBackbone(nn.Module):
  def __init__(self, base=32, factors=None, attention_levels=None,
               embedding_block=SinusoidalEmbedding,
               basic_block=basic_res_block,
               resample_block=BasicResample,
               attention_block=basic_attn_block,
               middle_block=MiddleBlock,
               dim=2, heads=8, cond_size=512, dropout=0.1):
    super().__init__()
    self.embedding = embedding_block(cond_size)
    self.blocks = nest_sequential(*([
      item
      for idx, subfactors in enumerate(factors)
      for item in [
        basic_block(
          base * in_factor, base * out_factor,
          dim=dim, cond_size=cond_size, dropout=dropout
        )
        for in_factor, out_factor in zip(
          subfactors[:-1], subfactors[1:]
        )
      ] + (
        [attention_block(base * subfactors[-1], heads=heads)]
        if idx in attention_levels else []
      ) + [
        resample_block(base * subfactors[-1], dim=dim)
      ]
    ] + [
      middle_block(base * factors[-1][-1], cond_size=cond_size)
    ]))

  def forward(self, inputs, time):
    time = self.embedding(time)
    return self.blocks(inputs, time)
