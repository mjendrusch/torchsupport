import os

import torch

from torchsupport.modules.structured import connected_entities as ce

class LazyNodes(object):
  def __init__(self, path, tensorize=lambda x: x, node_name="node"):
    self.properties = {}
    self.path = path
    self.node_name = node_name
    self.tensorize = tensorize
    with open(path) as nodefile:
      header = next(nodefile)
      properties = header.strip()[1:].split(";")
      for prop in properties:
        pparse = list(map(lambda x: x.strip(), prop.split("=")))
        self.properties[pparse[0]] = pparse[1]
    if 'length' not in self.properties:
      raise ValueError("Property missing: 'length'.")
    else:
      self.properties['length'] = int(self.properties['length'])

  def materialize(self, indices):
    print("IDC", indices)
    node_tensors = []
    with open(self.path) as nodefile:
      header = next(nodefile)
      for idx, line in enumerate(nodefile):
        if idx in indices:
          node_tensor = self.tensorize(line.strip().split(","))
          node_tensors.append(node_tensor.unsqueeze(0))
    return torch.cat(node_tensors, dim=0)

  def __len__(self):
    return self.properties['length']

class LazyAdjacency(object):
  def __init__(self, path):
    self.properties = {}
    self.path = path

  def materialize(self):
    return ce.AdjacencyStructure.from_csv(self.path)

  def __len__(self):
    return self.properties['length']
