from enum import Enum

from copy import copy, deepcopy

import torch
import networkx as nx

from torchsupport.data.collate import Collatable
from torchsupport.data.io import DeviceMovable
from torchsupport.structured.chunkable import (
  Chunkable, chunk_sizes, chunk_tensor
)

class MessageMode(Enum):
  iterative = 0
  constant = 1
  scatter = 2

class AbstractStructure(DeviceMovable, Chunkable, Collatable):
  message_modes = set()
  current_mode = None

  def mode(self, value):
    if value in self.message_modes:
      self.current_mode = value
      return self
    raise ValueError(
      f"Invalid MessageMode {value} not in {self.message_modes}."
    )

  def move_to(self, target):
    return self

  def mode_is(self, mode):
    if self.current_mode is not None:
      return mode == self.current_mode
    else:
      return mode in self.message_modes

  def message_iterative(self, *args, **kwargs):
    raise NotImplementedError("Abstract.")

  def message_constant(self, *args, **kwargs):
    raise NotImplementedError("Abstract.")

  def message_scatter(self, *args, **kwargs):
    raise NotImplementedError("Abstract.")

  def message_mode(self, mode, *args, **kwargs):
    to_call = getattr(self, f"message_{mode.name}")
    return to_call(*args, **kwargs)

  def message(self, *args, **kwargs):
    if self.current_mode is not None:
      return self.message_mode(self.current_mode, *args, **kwargs)
    if self.message_modes:
      the_mode = MessageMode(max(map(lambda x: x.value, self.message_modes)))
      return self.message_mode(the_mode, *args, **kwargs)
    raise ValueError("No valid MessageMode found.")

class ConnectionStructure(AbstractStructure):
  message_modes = {MessageMode.iterative}
  def __init__(self, source, target, connections):
    self.source = source
    self.target = target
    self.connections = connections
    self.lengths = [len(connections)]

  @classmethod
  def reachable_nodes(cls, start_nodes, structures, depth=1):
    if depth == 0:
      return start_nodes

    nodes = deepcopy(start_nodes)

    for typ in start_nodes:
      for structure in structures:
        if structure.source in start_nodes:
          if structure.target not in nodes:
            nodes[structure.target] = set([])
          for node in start_nodes[typ]:
            nodes[structure.target].update(set(structure.connections[node]))

    return cls.reachable_nodes(nodes, structures, depth=depth - 1)

  @classmethod
  def collate(cls, structures):
    assert structures
    assert all(map(lambda x: x.source == structures[0].source, structures))
    assert all(map(lambda x: x.target == structures[0].target, structures))
    connections = []
    offset = 0
    lengths = []
    for structure in structures:
      connections += list(map(
        lambda x: list(map(lambda y: y + offset, x)),
        structure.connections
      ))
      lengths.append(len(structure.connections))
      offset += len(structure.connections)
    result = cls(
      structures[0].source,
      structures[0].target,
      connections
    )
    result.lengths = lengths
    return result

  @classmethod
  def from_edges(cls, edges, source, target, nodes, directed=False):
    connections = [[] for _ in range(nodes)]
    for s, t in edges:
      connections[t].append(s)
      if not directed:
        connections[s].append(t)
    return cls(source, target, connections)

  @classmethod
  def from_nx(cls, graph, source, target):
    edge_list = [edge for edge in graph.edges]
    directed = isinstance(graph, nx.DiGraph)
    return cls.from_edges(edge_list, source, target, len(graph.nodes), directed=directed)

  @classmethod
  def from_csv(cls, path, source="nodes", target="nodes"):
    connections = []
    with open(path) as csv:
      for line in csv:
        cleaned = line.strip()
        if not cleaned:
          connections.append([])
        else:
          connections.append(list(map(int, cleaned.split(","))))
    return cls(source, target, connections)

  def move_to(self, device):
    return self

  def chunk(self, targets):
    sizes = chunk_sizes(self.lengths, len(targets))
    result = []
    offset = 0
    for size in sizes:
      the_copy = copy(self)
      the_copy.connections = [
        [
          item - offset
          for item in connection
        ]
        for connection in self.connections[offset:offset + size]
      ]
      result.append(the_copy)
      offset += size
    return result

  def select(self, sources, targets=None):
    """Selects a sub-adjacency structure from an adjacency structure
       given a set of source and target nodes to keep.

    Args:
      sources (list): list of source nodes to keep.
      targets (list or None): list of target nodes to keep. Defaults to sources.

    Returns:
      Subsampled adjacency structure containing only the desired nodes.
    """
    result = ConnectionStructure(self.source, self.target, [])
    if targets is None:
      targets = sources
    for target in targets:
      result.connections.append([
        sources.index(source)
        for source in self.connections[target]
        if source in sources
      ])
    return result

  def message(self, source, target):
    for idx, _ in enumerate(target):
      if self.connections[idx]:
        yield source[self.connections[idx]].unsqueeze(0)
      else:
        yield torch.zeros_like(source[0:1]).unsqueeze(0)

class CompoundStructure(AbstractStructure):
  message_modes = {MessageMode.iterative}
  current_mode = MessageMode.iterative
  def __init__(self, structures):
    assert all(map(lambda x: x.target == structures[0].target, structures))
    self.structures = structures

  @classmethod
  def collate(cls, structures):
    slots = [[] for idx in range(len(structures[0].structures))]
    for structure in structures:
      for idx, substructure in enumerate(structure.structures):
        slots[idx].append(substructure)
    for idx, slot in enumerate(slots):
      slots[idx] = ConnectionStructure.cat(slot)
    return cls(slots)

  def chunk(self, targets):
    result = [
      structure.chunk(targets)
      for structure in self.structures
    ]
    n_chunks = len(result[0])
    result = [
      CompoundStructure([
        structure[idx]
        for structure in result
      ])
      for idx in range(n_chunks)
    ]
    return result

  def message(self, source, target):
    for combination in zip(*map(lambda x: x.message(source, target), self.structures)):
      yield torch.cat(combination, dim=2)

class ConstantStructure(AbstractStructure):
  message_modes = {MessageMode.constant, MessageMode.iterative}
  current_mode = MessageMode.constant
  def __init__(self, source, target, connections):
    self.source = source
    self.target = target
    self.connections = connections
    if not isinstance(self.connections, torch.Tensor):
      self.connections = torch.tensor(
        connections, dtype=torch.long, requires_grad=False
      )
    self.lengths = [self.connections.size(0)]

  def move_to(self, device):
    result = self
    result.connections = result.connections.to(device)
    return result

  def chunk(self, targets):
    connections = []
    sizes = chunk_sizes(self.lengths, len(targets))
    step = len(self.lengths) // len(targets)
    offset = 0
    for idx, size in enumerate(sizes):
      the_connections = self.connections[offset:offset + size] - offset
      the_connections = the_connections.to(targets[idx])
      result = ConstantStructure(self.source, self.target, the_connections)
      result.lengths = self.lengths[step * idx:step * (idx + 1)]
      connections.append(result)
      offset += size
    return connections

  def update(self, data):
    (self.source, self.target, self.connections, self.lengths) = data
    return self

  def update_to(self, data):
    return copy(self).update(data)

  @classmethod
  def collate_parameters(cls, structures):
    assert structures
    assert all(map(lambda x: x.source == structures[0].source, structures))
    assert all(map(lambda x: x.target == structures[0].target, structures))
    connections = []
    lengths = []
    offset = 0
    for structure in structures:
      current_connections = structure.connections
      current_connections += offset
      connections.append(current_connections)
      lengths += structure.lengths
      offset += current_connections.size(0)
    source = structures[0].source
    target = structures[0].target
    connections = torch.cat(connections, dim=0)
    return source, target, connections, lengths

  @classmethod
  def collate(cls, structures):
    assert structures
    assert all(map(lambda x: x.source == structures[0].source, structures))
    assert all(map(lambda x: x.target == structures[0].target, structures))
    connections = []
    lengths = []
    offset = 0
    for structure in structures:
      current_connections = structure.connections
      current_connections += offset
      connections.append(current_connections)
      lengths += structure.lengths
      offset += current_connections.size(0)
    result = cls(
      structures[0].source,
      structures[0].target,
      torch.cat(connections, dim=0)
    )
    result.lengths = lengths
    print("coll lengths", lengths, len(structures), list(map(lambda x: x.lengths, structures)))
    return result

  def message(self, source, target):
    return source[self.connections]

class ConstantifiedStructure(ConstantStructure):
  def __init__(self, structure):
    self.source = structure.source
    self.target = structure.target
    self.connections = structure.connections
    self.structure = structure

  @classmethod
  def collate(cls, structures):
    structure_class = structures[0].structure.__class__
    return cls(structure_class.cat(structures))

  def chunk(self, targets):
    return [
      ConstantifiedStructure(the_chunk)
      for the_chunk in self.structure.chunk(targets)
    ]

  def message(self, source, target):
    results = []
    for message in self.structure.message(source, target):
      results.append(message)
    return torch.cat(results, dim=0)

class ConstantStructureMixin():
  @property
  def constant(self):
    return ConstantifiedStructure(self)

class ScatterStructure(AbstractStructure):
  message_modes = {MessageMode.scatter}
  current_mode = MessageMode.scatter
  def __init__(self, source, target, indices,
               connections, node_count=None):
    self.source = source
    self.target = target
    self.indices = indices
    self.connections = connections
    self.node_count = node_count
    if self.node_count is None:
      self.node_count = indices.max() + 1
    self.lengths = [len(self.indices)]
    self.node_counts = [self.node_count]

  @classmethod
  def from_connections(cls, source, target, connections):
    node_count = len(connections)
    indices = torch.tensor([
      item
      for idx, connection in enumerate(connections)
      for item in len(connection) * [idx]
    ], dtype=torch.long)
    connections = torch.tensor([
      item
      for connection in connections
      for item in connection
    ], dtype=torch.long)
    return cls(
      source, target,
      indices, connections,
      node_count=node_count
    )

  @classmethod
  def from_connection_structure(cls, structure):
    return cls.from_connections(
      structure.source, structure.target,
      structure.connections
    )

  def update(self, data):
    (self.source, self.target, self.indices, self.connections,
     self.node_count, self.node_counts, self.lengths) = data
    return self

  def update_to(self, data):
    return copy(self).update(data)

  @classmethod
  def collate_parameters(cls, structures):
    source = structures[0].source
    target = structures[0].target
    indices = []
    connections = []
    lengths = []
    node_counts = []
    offset = 0
    for struc in structures:
      indices.append(struc.indices + offset)
      connections.append(struc.connections + offset)
      offset += struc.node_count
      lengths += struc.lengths
      node_counts += struc.node_counts
    indices = torch.cat(indices, dim=0)
    connections = torch.cat(connections, dim=0)
    node_count = offset
    return (
      source, target, indices, connections,
      node_count, node_counts, lengths
    )

  @classmethod
  def collate(cls, structures):
    structure_class = structures[0].__class__
    source = structures[0].source
    target = structures[0].target
    indices = []
    connections = []
    lengths = []
    node_counts = []
    offset = 0
    for struc in structures:
      indices.append(struc.indices + offset)
      connections.append(struc.connections + offset)
      offset += struc.node_count
      lengths += struc.lengths
      node_counts += struc.node_counts
    result = structure_class(
      source, target,
      torch.cat(indices, dim=0),
      torch.cat(connections, dim=0)
    )
    result.node_count = offset
    result.node_counts = node_counts
    result.lengths = lengths
    return result

  def chunk(self, targets):
    sizes = chunk_sizes(self.lengths, len(targets))
    step = len(self.lengths) // len(targets)
    result = []
    offset = 0
    index_offset = 0
    for idx, size in enumerate(sizes):
      the_copy = copy(self)
      the_copy.indices = self.indices[offset:offset + size] - index_offset
      the_copy.connections = self.connections[offset:offset + size] - index_offset
      the_copy.lengths = self.lengths[idx * step:(idx + 1) * step]
      the_copy.node_counts = self.node_counts[idx * step:(idx + 1) * step]
      the_copy.node_count = sum(the_copy.node_counts)
      result.append(the_copy)
      offset += size
      index_offset += the_copy.node_count
    return result

  def __len__(self):
    return self.node_count

  def __getitem__(self, idx):
    assert idx < self.node_count
    return self.connections[self.indices == idx]

  def message(self, source, target):
    return source[self.connections], target[self.indices], self.indices, self.node_count

class SubgraphStructure(AbstractStructure):
  message_modes = {MessageMode.iterative, MessageMode.scatter}
  current_mode = MessageMode.scatter
  def __init__(self, membership):
    self.indices = membership
    unique, counts = self.indices.unique(return_counts=True)
    self.unique = unique
    self.counts = counts

  @classmethod
  def collate(cls, structures):
    structure_class = structures[0].__class__
    indices = []
    offset = 0
    for struc in structures:
      indices.append(struc.indices + offset)
      offset += struc.unique.max() + 1
    result = structure_class(torch.cat(indices, dim=0))
    return result

  def chunk(self, targets):
    sizes = chunk_sizes(self.counts, len(targets))
    result = []
    offset = 0
    for size in sizes:
      the_copy = copy(self)
      the_copy.indices = self.indices[offset:offset + size]
      the_copy.indices = the_copy.indices - the_copy.indices[0]
      the_copy.unique, the_copy.counts = the_copy.indices.unique(return_counts=True)
      result.append(the_copy)
      offset += the_copy.indices.size(0)
    return result

  def message_iterative(self, source, target):
    for subgraph in self.unique:
      index = (self.indices == subgraph).view(-1).nonzero()
      yield source[index]

  def message_scatter(self, source, target):
    return source, self.indices
