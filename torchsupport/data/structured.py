import os
import random

import torch
from torch.utils.data import Dataset

from torchsupport.modules.structured import connected_entities as ce
from torchsupport.data.graphio import LazyNodes, LazyAdjacency

class LazySubgraphDataset(Dataset):
  def __init__(self, path, node_name, edge_names, depth=3):
    self.depth = depth
    self.path = path
    self.node_name = node_name
    self.nodes = []
    self.adjacencies = []
    for root, _, names in os.walk(path):
      for name in names:
        if name.endswith(f"{node_name}.node"):
          base = ".".join(name.split(".")[:-2])
          self.nodes.append(LazyNodes(os.path.join(root, f"{base}.{node_name}.node")))
          self.adjacencies.append({
            name: LazyAdjacency(os.path.join(root, f"{base}.{name}.struct"))
            for name in edge_names
          })

  def __len__(self):
    return len(self.nodes)

  def __getitem__(self, idx):
    start_node = random.randint(0, len(self.nodes[idx]))
    adjacencies = [
      self.adjacencies[idx][name].materialize()
      for name in self.adjacencies[idx]
    ]
    reachable = ce.ConnectionStructure.reachable_nodes(
      {self.node_name: set([start_node])},
      adjacencies,
      depth=self.depth
    )
    reachable = list(reachable[self.node_name])
    node_tensor = self.nodes[idx].materialize(reachable)
    adjacencies = [
      adj.select(reachable)
      for adj in adjacencies
    ]
    return node_tensor, adjacencies
