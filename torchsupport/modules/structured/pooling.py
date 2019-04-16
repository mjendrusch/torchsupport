import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.structured import connected_entities as ce
from torchsupport.modules.structured import entitystructure as es

class DeletionPool(nn.Module):
  def __init__(self, size):
    super(DeletionPool, self).__init__()
    self.project = nn.Linear(size, 1)

  def forward(self, data, structure):
    vals = self.project(data)
    median_val = torch.median(vals)
    keep_nodes = (vals > median_val).nonzeros()
    pooled_data = vals[keep_nodes] * data[keep_nodes]
    pooled_structure = es.ConnectMissing(structure, keep_nodes)
    return pooled_data, pooled_structure

class SelectionPool(nn.Module):
  pass

class CliquePool(nn.Module):
  def __init__(self):
    super(CliquePool, self).__init__()
