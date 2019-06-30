import random
from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as func

from .connection import (
  AbstractStructure, ConnectionStructure,
  SubgraphStructure, ConstantStructureMixin,
  ScatterStructure, MessageMode
)

def to_graph(batched_tensor):
  squashed_tensor = batched_tensor.view(
    batched_tensor.size(0), batched_tensor.size(1), -1
  )
  batch_size = squashed_tensor.size(0)
  length = squashed_tensor.size(2)

  graph_tensor = batched_tensor.view(-1, batched_tensor.size(1))
  subgraph_structure = SubgraphStructure([
    range(idx * length, (idx + 1) * length)
    for idx in range(batch_size)
  ])

  return graph_tensor, subgraph_structure

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
  substructure = ConnectionStructure(
    structure.source,
    structure.target,
    connections
  )
  return full_nodes, substructure

class DropoutStructure(AbstractStructure):
  def __init__(self, structure, p=0.5):
    """Drops random messages from a given structure.

    Args:
      structure (ConnectionStructure or ScatterStructure): Structure to sample edges from.
      p (float): probability of dropping an edge.
    """
    super(DropoutStructure, self).__init__()
    self.structure = structure
    self.message_modes.update(self.structure.message_modes)
    self.p = p

  def message_scatter(self, source, target):
    indices = self.structure.indices
    connections = self.structure.connections
    total = indices.size(0)
    keep, _ = torch.randperm(total)[:int((1 - self.p) * total)].sort()
    indices = indices[keep]
    connections = indices[keep]
    return source[connections], target[indices], indices

  def message_iterative(self, source, target):
    for msg in self.structure.message(source, target):
      randoms = torch.rand(len(msg)) > self.p
      if randoms.sum() > 0:
        yield msg[:, randoms]
      else:
        yield torch.zeros_like(msg[0].unsqueeze(0))

  def message(self, source, target):
    if MessageMode.scatter in self.message_modes:
      return self.message_scatter(source, target)
    if MessageMode.iterative in self.message_modes:
      for message in self.message_iterative(source, target):
        yield message

class NHopStructure(ConnectionStructure):
  def __init__(self, structure, depth, with_self=False):
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

class InverseStructure(ConnectionStructure):
  def __init__(self, structure, source_size=None):
    if source_size is None:
      source_size = max([
        item
        for item in connection
        for connection in structure.connections
      ]) + 1
    super(InverseStructure, self).__init__(
      structure.target,
      structure.source,
      [
        [
          idy
          for idy, connection in enumerate(structure.connections)
          if idx in connection
        ]
        for idx in range(source_size)
      ]
    )

class EdgeStructure(ConnectionStructure):
  def __init__(self, structure, source_size=None):
    self.structure = structure
    self.inverse_structure = InverseStructure(
      structure,
      source_size=source_size
    )
    super(EdgeStructure, self).__init__(
      structure.source,
      structure.source,
      [
        list({
          second_connection
          for second_connection in self.inverse_structure.connections[item]
          for item in connection
        })
        for idx, connection in enumerate(self.structure.connections)
      ]
    )

  def message(self, source, target):
    node, edge = source
    for idx, _ in enumerate(target):
      if self.inverse_structure.connections[idx]:
        yield (
          node[self.connections[idx]].unsqueeze(0),
          edge[self.inverse_structure.connections[idx]].unsqueeze(0)
        )
      else:
        yield (
          torch.zeros_like(node[0:1]).unsqueeze(0),
          torch.zeros_like(edge[0:1]).unsqueeze(0)
        )

def _connect_missing_aux(connection, structure, keep_nodes, depth):
  if depth == 0:
    return
  next_depth = depth - 1 if depth is not None else None
  for node in connection:
    if node in keep_nodes:
      yield node
    else:
      connected_nodes = structure.connection[node]
      for connected in _connect_missing_aux(
          connected_nodes, structure, keep_nodes, next_depth
      ):
        yield connected

class ConnectMissing(ConnectionStructure):
  def __init__(self, structure, keep_nodes, depth=None):
    assert structure.source == structure.target
    super(ConnectMissing, self).__init__(
      structure.source,
      structure.target,
      [
        [
          node
          for node in _connect_missing_aux(
            connection, structure, keep_nodes, depth
          )
        ]
        for idx, connection in enumerate(structure.connections)
        if idx in keep_nodes
      ]
    )

class PairwiseData(ConstantStructureMixin, AbstractStructure):
  """Auxiliary structure for attaching pairwise "edge" data
  given data local to sources and targets.
  """
  def __init__(self, structure):
    """Auxiliary structure for attaching pairwise "edge" data
    given data local to sources and targets.

    Args:
      structure (AbstractStructure): underlying connection structure.
      source_data (torch.Tensor): data present at source nodes.
      target_data (torch.Tensor): data present at target nodes.
    """
    super(PairwiseData, self).__init__()
    self.structure = structure
    self.message_modes.update(self.structure.message_modes)
    self.current_mode = self.structure.current_mode

  def compare(self, source, target):
    """Compare source and target data to produce edge features.

    Args:
      source (torch.Tensor): (connections, source_features, size) tensor of source data.
      target (torch.Tensor): (target_features, size) tensor of target data.

    Returns:
      Edge data tensor of size (connections, edge_features, size) tensor of derived edge data.
    """
    raise NotImplementedError("Abstract.")

  def compare_empty(self):
    """Return empty comparison.

    Returns:
      Default tensor for absent edge data.
    """
    raise NotImplementedError("Abstract.")

  def message_iterative(self, source, target):
    for idx, _ in enumerate(target):
      neighbours = self.structure.connections[idx]
      has_elements = len(neighbours) > 0
      if has_elements:
        pairwise = self.compare(
          source[self.structure.connections[idx]],
          target[idx]
        )
        yield pairwise.unsqueeze(0)
      else:
        pairwise = self.compare_empty()
        yield pairwise.unsqueeze(0)

  def message_scatter(self, source, target):
    comparison = self.compare(
      source[self.structure.connections],
      target[self.structure.indices]
    )
    return comparison, self.structure.indices

  def message_constant(self, source, target):
    packed_source = source[self.structure.connections]
    packed_target = target.unsqueeze(1).expand(
      target.size(0), packed_source.size(1), *target.shape[1:]
    )
    unpacked_source = packed_source.view(-1, *packed_source.shape[2:])
    unpacked_target = packed_target.contiguous().view(-1, *packed_target.shape[2:])
    comparison = self.compare(unpacked_source, unpacked_target)
    packed_comparison = comparison.view(
      -1, self.structure.connections.size(1), *comparison.shape[1:]
    )
    return packed_comparison

class PairwiseStructure(PairwiseData):
  """Auxiliary structure for attaching pairwise "edge" data
  given data local to sources and targets.
  """
  def __init__(self, structure, source_data, target_data):
    """Auxiliary structure for attaching pairwise "edge" data
    given data local to sources and targets.

    Args:
      structure (ConnectionStructure): underlying connection structure.
      source_data (torch.Tensor): data present at source nodes.
      target_data (torch.Tensor): data present at target nodes.
    """
    super(PairwiseStructure, self).__init__(structure)
    self.source_data = source_data
    self.target_data = target_data

  def message_iterative(self, source, target):
    for idx, _ in enumerate(target):
      if self.structure.connections[idx]:
        data = source[self.structure.connections[idx]]
        pairwise = self.compare(
          self.source_data[self.structure.connections[idx]],
          self.target_data[idx]
        )
        yield torch.cat((data, pairwise), dim=1).unsqueeze(0)
      else:
        data = torch.zeros_like(source[0:1])
        pairwise = self.compare_empty()
        yield torch.cat((data, pairwise), dim=1).unsqueeze(0)

  def message_constant(self, source, target):
    pairwise = super().message_constant(self.source_data, self.target_data)
    data = self.structure.message_constant(source, target)
    return torch.cat((data, pairwise))

  def message_scatter(self, source, target):
    pairwise, indices = super().message(self.source_data, self.target_data)
    source, target, indices = self.structure.message_scatter(source, target)
    return torch.cat((source, pairwise), dim=1), target, indices

class DistanceStructure(ConnectionStructure):
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
        for node in range(len(entity_tensor))
      ]
    )

  @staticmethod
  def _traverse_aux(entity_tensor, subgraph_structure, entity, radius, metric):
    accept_nodes = []
    graph_slice = subgraph_structure.membership[entity]
    distances = metric(
      entity_tensor[graph_slice] - entity_tensor[entity]
    )
    for node, distance in enumerate(distances, graph_slice.start):
      if distance < radius:
        accept_nodes.append(node)
    return accept_nodes

class ImplicitDistanceStructure(ConnectionStructure):
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
        ), dim=1).unsqueeze(0)
      else:
        yield torch.zeros(1, source.size(1) + 1).unsqueeze(0)

class OffsetOrientationStructure(ConnectionStructure):
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
        ), dim=1).unsqueeze(0)
      else:
        yield source[0:0].unsqueeze(0)

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
