import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.structured import structures as cs

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
