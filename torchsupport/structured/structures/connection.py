from copy import copy, deepcopy

import torch
import networkx as nx

from torchsupport.data.collate import Collatable
from torchsupport.data.io import DeviceMovable
from torchsupport.structured.chunkable import (
  Chunkable, chunk_sizes, chunk_tensor
)

class RangedArray(object):
  def __init__(self, values, ranges):
    self.values = values
    self.ranges = ranges

  def __len__(self):
    return len(self.ranges) - 1

  def __getitem__(self, idx):
    return self.values[self.ranges[idx]:self.ranges[idx]+1]

class ConnectionStructure(DeviceMovable, Collatable, object):
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

class CompoundStructure(ConnectionStructure):
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

class SubgraphStructure(ConnectionStructure):
  def __init__(self, membership):
    self.membership = membership

  @classmethod
  def collate(cls, structures):
    return cls([
      subgraph
      for structure in structures
      for subgraph in structure.membership
    ])

  def message(self, source, target):
    for subgraph in self.membership:
      yield source[subgraph]

class ConstantStructure(ConnectionStructure):
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
    return ConstantStructure(
      self.source, self.target,
      self.connections.to(device)
    )

  def chunk(self, targets):
    connections = []
    sizes = chunk_sizes(self.lengths, len(targets))
    offset = 0
    for idx, size in enumerate(sizes):
      the_connections = self.connections[offset:offset + size] - offset
      the_connections = the_connections.to(targets[idx])
      connections.append(the_connections)
      offset += size
    return [
      ConstantStructure(self.source, self.target, the_chunk)
      for the_chunk in connections
    ]

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

class ScatterStructure(ConnectionStructure):
  def __init__(self, source, target, indices,
               connections, length=None):
    self.source = source
    self.target = target
    self.indices = indices
    self.connections = connections
    self.lengths = [
      length if length is not None else indices.max()
    ]

  @classmethod
  def from_connections(cls, source, target, connections):
    indices = torch.Tensor([
      item
      for idx, connection in enumerate(connections)
      for item in len(connection) * [idx]
    ])
    connections = torch.Tensor([
      connection
      for connection in connections
      for item in connection
    ])
    return cls(
      source, target,
      indices, connections,
      length=len(connections)
    )

  @classmethod
  def collate(cls, structures):
    structure_class = structures[0].structure.__class__
    source = structures[0].source
    target = structures[0].target
    indices = []
    connections = []
    lengths = []
    offset = 0
    for struc in structures:
      indices.append(struc.indices + offset)
      connections.append(struc.connections + offset)
      offset += sum(struc.lengths)
      lengths += struc.lengths
    lengths = [
      length
      for struc in structures
      for length in struc.lengths
    ]
    result = structure_class(
      source, target,
      torch.cat(indices, dim=0),
      torch.cat(connections, dim=0)
    )
    result.lengths = lengths
    return result

  def chunk(self, targets):
    sizes = chunk_sizes(self.lengths, len(targets))
    result = []
    offset = 0
    for size in sizes:
      the_copy = copy(self)
      the_copy.connections = self.connections[offset:offset + size]
      result.append(the_copy)
      offset += size
    return result

  def message(self, source, target):
    return source[self.connections], target[self.indices], self.indices
