import random
from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.structured.connected_entities import ConnectionStructure, AdjacencyStructure

def random_substructure(structure, num_nodes, depth):
  initial_nodes = random.sample(
    range(len(structure.connection)), num_nodes
  )
  full_nodes = copy(initial_nodes)
  for step in range(depth):
    connected_nodes = []
    for node in initial_nodes:
      connected_nodes += structure.connections[node]
    node_set = set(connected_nodes)
    initial_nodes = node_set - set(full_nodes)
    full_nodes += list(set(connected_nodes))
  full_nodes.sort()
  connections = list(map(
    lambda x: list(map(
      lambda y: full_nodes.index(y)
    ))
  ))
  substructure = AdjacencyStructure(
    structure.source,
    structure.target,
    connections
  )
  return full_nodes, substructure

class DropoutStructure(ConnectionStructure):
  def __init__(self, structure, p=0.5):
    """Drops random messages from a given structure.

    Args:
      structure (ConnectionStructure): Structure to sample edges from.
      p (float): probability of dropping an edge.
    """
    super(DropoutStructure, self).__init__(
      structure.source,
      structure.target,
      structure.connections
    )
    self.p = p

  def message(self, source, target):
    print(source, target)
    for msg in super(DropoutStructure, self).message(source, target):
      randoms = torch.rand(len(msg)) > self.p
      if randoms.sum() > 0:
        yield msg[randoms]
      else:
        yield torch.zeros_like(msg[0].unsqueeze(0))

class NHopStructure(AdjacencyStructure):
  def __init__(self, structure, depth, with_self=True):
    """Computes a standard node traversal for a given node.

    Args:
      depth (int): number of hops in the n-hop traversal.
    """
    assert structure.source == structure.target
    super(NHopStructure, self).__init__(
      structure.source,
      structure.target,
      [
        NHopStructure._traverse_aux(
          structure.connections, node,
          with_self, depth
        )
        for node in range(len(structure.connections))
      ]
    )

  @staticmethod
  def _traverse_aux(connections, entity, with_self, depth):
    if depth == 0:
      return [entity] if with_self else []
    nodes = [entity] if with_self else []
    edges = connections[entity]
    nodes += edges
    for new_node in edges:
      if new_node != entity:
        new_nodes = NHopStructure._traverse_aux(
          connections, new_node, with_self, depth - 1
        )
        nodes += new_nodes
    nodes = list(set(nodes))
    return nodes

class DistanceStructure(AdjacencyStructure):
  def __init__(self, entity_tensor, subgraph_structure, typ,
               radius=1.0, metric=lambda x, y: torch.norm(x - y, 2)):
    """Computes a traversal of nodes within a given distance of a starting node.

    Args:
      reject_depth (int): nodes connected to the origin node up to this depth
        are discarded.
      radius (int): nodes within this distance are accepted.
      position_slice (slice): slice of input features to be used for distance.

    Returns:
      Nodes within a given `radius` from the origin node, at least `reject_depth`
      hops away from the origin node.
    """
    super(DistanceStructure, self).__init__(
      typ, typ,
      [
        DistanceStructure._traverse_aux(
          entity_tensor, subgraph_structure,
          node, radius, metric
        )
        for node in range(len(entity_tensor.current_view))
      ]
    )

  @staticmethod
  def _traverse_aux(entity_tensor, subgraph_structure, entity, radius, metric):
    accept_nodes = []
    graph_slice = subgraph_structure.membership[entity]
    distances = metric(
      entity_tensor.current_view[graph_slice] - entity_tensor.current_view[entity]
    )
    for node, distance in enumerate(distances, graph_slice.start):
      if distance < radius:
        accept_nodes.append(node)
    return accept_nodes

class ImplicitDistanceStructure(AdjacencyStructure):
  def __init__(self, position, structure):
    """Structure implicitly providing a relative distance."""
    super(ImplicitDistanceStructure, self).__init__(
      structure.source,
      structure.target,
      structure.connections
    )
    self.position = position

  def message(self, source, target):
    for idx, _ in enumerate(target):
      if self.connections[idx]:
        offset = self.position[self.connections[idx]] - self.position[idx:idx + 1]
        yield torch.cat((
          offset.norm(dim=1, keepdim=True),
          source[self.connections[idx]]
        ), dim=1)
      else:
        yield torch.zeros(1, source.size(1) + 1)

class OffsetOrientationStructure(AdjacencyStructure):
  def __init__(self, position, orientation, structure):
    """Structure implicitly providing a relative offset."""
    super(OffsetOrientationStructure, self).__init__(
      structure.source,
      structure.target,
      structure.connections
    )
    self.position = position
    self.orientation = orientation

  def message(self, source, target):
    for idx, _ in enumerate(target):
      if self.connections[idx]:
        offset = self.position[self.connections[idx]] - self.position[idx:idx + 1]
        rotation = _make_rotation(-self.orientation[idx])
        offset = torch.matmul(rotation, offset)
        yield torch.cat((
          offset, source[self.connections[idx]]
        ), dim=1)
      else:
        yield source[0:0]

def _make_rotation(angles):
  rot_0 = _make_rotation_idx(angles[0], 0)
  rot_1 = _make_rotation_idx(angles[1], 1)
  rot_2 = _make_rotation_idx(angles[2], 2)
  return torch.matmul(rot_0, torch.matmul(rot_1, rot_2))

def _make_rotation_idx(angles, idx):
  result = torch.zeros(3, 3)
  for i in range(3):
    result[i, i] = 1
  result[(idx + 1) % 3, (idx + 1) % 3] = angles.cos()
  result[(idx + 2) % 3, (idx + 2) % 3] = angles.cos()
  result[(idx + 1) % 3, (idx + 2) % 3] = angles.sin()
  result[(idx + 2) % 3, (idx + 1) % 3] = -angles.sin()
  return result
