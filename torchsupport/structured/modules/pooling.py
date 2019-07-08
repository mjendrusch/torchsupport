import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.structured import structures as cs
from torchsupport.structured import scatter

class DeletionPool(nn.Module):
  def __init__(self, size):
    super(DeletionPool, self).__init__()
    self.project = nn.Linear(size, 1)

  def forward(self, data, structure):
    vals = self.project(data)
    median_val = torch.median(vals)
    keep_nodes = (vals > median_val).nonzeros()
    pooled_data = vals[keep_nodes] * data[keep_nodes]
    pooled_structure = cs.ConnectMissing(structure, keep_nodes)
    return pooled_data, pooled_structure

class SelectionPool(nn.Module):
  pass

class CliquePool(nn.Module):
  def __init__(self):
    super(CliquePool, self).__init__()

class GraphPool(nn.Module):
  def __init__(self):
    super(GraphPool, self).__init__()

  def combine(self, nodes, indices):
    raise NotImplementedError("Abstract.")

  def forward(self, nodes, indices):
    return self.combine(nodes, indices)

class MILAttention(GraphPool):
  def __init__(self, in_size, out_size, attention_size, heads):
    super(MILAttention, self).__init__()
    self.heads = heads
    self.query = nn.Linear(attention_size, heads)
    self.gate = nn.Linear(in_size, attention_size)
    self.key = nn.Linear(in_size, attention_size)
    self.value = nn.Linear(in_size, heads * attention_size)
    self.out = nn.Linear(heads * attention_size, out_size)

  def combine(self, nodes, indices):
    logits = self.query(torch.tanh(self.key(nodes)) * torch.sigmoid(self.gate(nodes)))
    weight = scatter.softmax(logits, indices)
    value = self.value(nodes)
    weight = weight.view(*weight.shape[:-1], 1, self.heads)
    value = value.view(*value.shape[:-1], -1, self.heads)
    result = scatter.add(weight * value, indices)
    return self.out(result.view(*result.shape[:-2], -1))

class MILRouting(GraphPool):
  def __init__(self, in_size, k=3):
    super(MILRouting, self).__init__()
    self.k = k

  def forward(self, nodes, indices):
    weights = torch.zeros(nodes.size(0), 1)
    for idx in range(self.k):
      smax = torch.softmax(weights, dim=0)
      sigma = scatter.add(smax * nodes)
      norm = torch.norm(sigma, dim=1, keepdim=True)
      norm2 = norm ** 2
      sval = sigma / norm * norm2 / (1 + norm2)
      weights = weights + nodes * sigma[indices]
    return sigma
