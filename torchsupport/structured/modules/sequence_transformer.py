import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.structured.scatter import pad, unpad

def dot_attention(x, y):
  return (x * y * torch.tensor(x.shape[-1]).rsqrt()).sum(dim=-1)

def l2_attention(x, y):
  return (x - y).norm(dim=-1)

def linear_attention(x, y):
  return (x - y).mean(dim=-1)

def elu_features(x):
  return func.elu(x) + 1

class SequenceMultiHeadAttention(nn.Module):
  r"""Implements multi-head attention for a non-structured sets.
  Computes multi-head attention on a ragged tensor.

  Args:
    in_size (int): size of the input embedding.
    out_size (int): size of the output embedding.
    hidden_size (int): size of the internal embedding.
    attention_size (int): size of the query and key vectors.
    heads (int): number of concurrent attention heads.
    similarity (callable): kernel comparing key and query vectors.

  Shape:
    - Inputs: :math:`(\sum_i N_i, F_{in})`
    - Index: :math:`\sum_i N_i`
    - Outputs: :math:`(\sum_i N_i, F_{out})`
  """
  def __init__(self, in_size, out_size,
               hidden_size=128, attention_size=128,
               heads=8, similarity=None):
    super().__init__()
    self.attention_size = attention_size
    self.hidden_size = hidden_size
    self.heads = heads
    self.sim = similarity or dot_attention
    self.key = nn.Linear(in_size, attention_size * heads, bias=False)
    self.query = nn.Linear(in_size, attention_size * heads, bias=False)
    self.value = nn.Linear(in_size, hidden_size * heads, bias=False)
    self.out = nn.Linear(hidden_size * heads, out_size, bias=False)

  def forward(self, inputs, index):
    key = self.key(inputs).view(inputs.size(0), self.heads, self.attention_size)
    query = self.query(inputs).view(inputs.size(0), self.heads, self.attention_size)
    value = self.value(inputs).view(inputs.size(0), self.heads, self.hidden_size)
    key, _, indices, _ = pad(key, index)
    query, _, indices, _ = pad(query, index)
    value, _, indices, _ = pad(value, index)
    sim = self.sim(key[:, :, None], query[:, None, :]).unsqueeze(-1)

    # mask bad values
    mask = torch.ones_like(sim, dtype=torch.bool)
    mask.view(-1, *mask.shape[2:])[indices] = False
    sim[mask] = -float("inf")
    sim = sim.softmax(dim=1)

    result = (value[:, :, None] * sim).sum(dim=1)
    result = result.view(*result.shape[:-2], -1)
    result = unpad(result, indices)
    result = self.out(result)
    return result

class SequenceLinearAttention(SequenceMultiHeadAttention):
  r"""Implements linear-order multi-head attention for a non-structured sets.
  Computes multi-head attention on a ragged tensor.

  Args:
    in_size (int): size of the input embedding.
    out_size (int): size of the output embedding.
    hidden_size (int): size of the internal embedding.
    attention_size (int): size of the query and key vectors.
    heads (int): number of concurrent attention heads.
    similarity (callable): features to apply to query and key vectors.

  Shape:
    - Inputs: :math:`(\sum_i N_i, F_{in})`
    - Index: :math:`\sum_i N_i`
    - Outputs: :math:`(\sum_i N_i, F_{out})`
  """
  def __init__(self, *args, similarity=None, **kwargs):
    super().__init__(
      *args, similarity=similarity or elu_features, **kwargs
    )

  def forward(self, inputs, index):
    key = self.key(inputs).view(inputs.size(0), self.heads, self.attention_size)
    query = self.query(inputs).view(inputs.size(0), self.heads, self.attention_size)
    value = self.value(inputs).view(inputs.size(0), self.heads, self.hidden_size)
    key, _, indices, _ = pad(self.sim(key), index)
    query, _, indices, _ = pad(self.sim(query), index)
    value, _, indices, _ = pad(value, index)

    # (N, heads, D, V)
    key_value = (key[:, :, :, :, None] * value[:, :, :, None, :]).sum(dim=1)
    # (N, heads, D)
    sum_key = key.sum(dim=1)
    # (N, L, heads, V)
    query_key_value = (query[:, :, :, :, None] * key_value[:, None, :, :, :]).sum(dim=2)
    # (N, L, heads, 1)
    query_sum_key = (query * sum_key[:, None, :, :]).sum(dim=-1).unsqueeze(-1)
    result = query_key_value / (query_sum_key + 1e-6)

    result = result.view(result.shape[:2], -1)
    result = unpad(result, indices)
    result = self.out(result)
    return result
