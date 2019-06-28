from copy import copy, deepcopy

import torch
import networkx as nx

from torchsupport.data.collate import Collatable
from torchsupport.data.io import DeviceMovable
from torchsupport.modules.structured.parallel import (
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
      the_copy.connections = self.connections[offset:offset + size]
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

  def move_to(self, device):
    return ConstantStructure(
      self.source, self.target,
      self.connections.to(device)
    )

  def chunk(self, targets):
    sizes = chunk_sizes(self.lengths, len(targets))
    connections = chunk_tensor(self.connections, sizes, targets)
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
    offset = 0
    for structure in structures:
      current_connections = structure.connections
      current_connections += offset
      connections.append(current_connections)
      offset += current_connections.size(0)
    return cls(
      structures[0].source,
      structures[0].target,
      torch.cat(connections, dim=0)
    )

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
  def __init__(self, source, target, connections):
    self.source = source
    self.target = target
    self.indices = torch.Tensor([
      item
      for idx, connection in enumerate(connections)
      for item in len(connection) * [idx]
    ])
    self.connections = torch.Tensor([
      connection
      for connection in connections
      for item in connection
    ])

  def message(self, source, target):
    return source[self.connections], target[self.indices], self.indices

class AdjacencyStructure(ConnectionStructure):
  def __init__(self, source, target, connections):
    super(AdjacencyStructure, self).__init__(source, target, connections)
    self._laplacian = None
    self._adjacency_matrix = None
    self._recompute_laplacian = True
    self._recompute_adjacency_matrix = True

  def _laplacian_element(self, i, j):
    """Laplacian element.

    Args:
      i, j (int): position for element calculation.

    Returns:
      Element of the graph Laplacian at the subscript `i, j`.
    """
    if i == j:
      return len(self.connections[i])
    else:
      return -int(j in self.connections[i])

  def _compute_laplacian(self):
    """Precomputes the graph Laplacian."""
    self._recompute_laplacian = False
    indices = [
      (node, edge)
      for node, edges in enumerate(self.connections)
      for edge in edges + [node]
    ]
    values = torch.zeros(len(indices))
    for idx, index in enumerate(indices):
      values[idx] = self._laplacian_element(*index)
    indices = torch.Tensor(indices).t()
    self._laplacian = torch.sparse_coo_tensor(
      indices, values,
      (len(self.connections), len(self.connections))
    )

  def _decompute_laplacian(self):
    """Decomputes the graph Laplacian."""
    self._laplacian = None
    self._recompute_laplacian = True

  def laplacian_action(self, vector, matrix_free=True, normalized=False):
    """Computes the action of the graph Laplacian on a `Tensor`.

    Args:
      vector (Tensor): one-dimensional `Tensor` upon which the
        Laplacian shall act.
      matrix_free (bool): compute the Laplacian action without
        materializing the Laplacian?
    """
    if matrix_free:
      out = torch.zeros_like(vector)
      for node, edges in enumerate(self.connections):
        out[node] = len(edges) * vector[node] - vector[edges].sum(dim=0)
        if normalized:
          out[node] /= len(edges)
      return out
    if self._recompute_laplacian:
      self._compute_laplacian()
    out = self._laplacian.mm(vector)
    if normalized:
      norm = torch.Tensor([len(edges) for edges in self.connections])
      out /= norm
    return out

  def _compute_adjacency_matrix(self):
    """Computes the graph adjacency matrix."""
    self._recompute_adjacency_matrix = False
    indices = torch.Tensor([
      (node, edge)
      for node, edges in enumerate(self.connections)
      for edge in edges + [node]
    ]).t()
    values = torch.ones(indices.size(1))
    self._adjacency_matrix = torch.sparse_coo_tensor(
      indices, values,
      (len(self.connections), len(self.connections))
    )

  def _decompute_adjacency_matrix(self):
    """Decomputes the graph adjacency matrix."""
    self._recompute_adjacency_matrix = True
    self._adjacency_matrix = None

  def adjacency_action(self, vector, matrix_free=True):
    """Computes the action of the graph adjacency matrix on a `Tensor`.

    Args:
      vector (Tensor): one-dimensional `Tensor` upon which the
        adjacency matrix shall act.
      matrix_free (bool): compute the Laplacian action without
        materializing the adjacency matrix?
    """
    if matrix_free:
      out = torch.zeros_like(vector)
      for node, edges in enumerate(self.connections):
        out[node] = vector[edges].sum(dim=0)
      return out
    else:
      if self._recompute_adjacency_matrix:
        self._compute_adjacency_matrix()
      return self._adjacency_matrix(vector)

class EntityTensor(object):
  """Node-only graph tensor."""

  def __init__(self, **entity_tensors):
    """Node-only graph tensor."""
    self._entity_types = []
    self._entity_view = None
    for typ in entity_tensors:
      self._entity_types.append(typ)
      setattr(self, typ, entity_tensors[typ])

  @classmethod
  def collate(cls, tensors, view=None, dim=0):
    assert tensors
    result = tensors[0].new_like()
    if view is not None:
      for typ in result.entity_types:
        setattr(
          result, typ,
          torch.cat(list(map(
            lambda x: getattr(x, typ),
            tensors
          )), dim=dim)
        )
    else:
      result.view(view)
      result.current_view = torch.cat(list(map(
        lambda x: getattr(x, view),
        tensors
      )), dim=dim)
    return result

  @property
  def entity_types(self):
    return {
      typ : getattr(self, typ).size()[1:]
      for typ in self._entity_types
    }

  def __repr__(self):
    class_name = self.__class__.__name__
    return f"{class_name}({self.entity_types})"

  def new_like(self):
    result = copy(self)
    result._entity_types = list(result._entity_types)
    return result

  @property
  def current_view(self):
    return getattr(self, self._entity_view)

  @current_view.setter
  def current_view(self, data):
    setattr(self, self._entity_view, data)

  def view(self, name):
    result = copy(self)
    result._entity_view = name
    return result

  def subset(self, **kwargs):
    result = copy(self)
    for key in kwargs:
      setattr(result, key, getattr(result, key)[kwargs[key]])
    return result

  def add_entity(self, typ, tensor):
    self._entity_types.append(typ)
    setattr(self, typ, tensor)
    return self

  def splice(self, target, **kwargs):
    result = self.new_like()
    for key in kwargs:
      result.add_entity(target, getattr(result, key)[kwargs[key]])
      break
    return result

def _gen_placeholder_arithmetic(operation):
  def _placeholder_arithmetic(self, other):
    if not isinstance(other, EntityTensor):
      tmp = self.new_like()
      for typ in self.entity_types:
        setattr(tmp, typ, other)
      other = tmp
    if self._entity_view is not None and other._entity_view is None:
      result = copy(other)
      for typ in self.entity_types:
        setattr(
          result, typ,
          getattr(self.current_view, operation)(getattr(other, typ))
        )
    if other._entity_view is not None and self._entity_view is None:
      result = copy(self)
      for typ in self.entity_types:
        setattr(
          result, typ,
          getattr(getattr(self, typ), operation)(other.current_view)
        )
    elif other._entity_view is not None and self._entity_view is not None:
      result = copy(self)
      result.current_view = getattr(self.current_view, operation)(other.current_view)
    else:
      result = copy(self)
      for typ in self.entity_types:
        setattr(
          result, typ,
          getattr(getattr(self, typ), operation)(getattr(other, typ))
        )
    return result
  return _placeholder_arithmetic

for _operation in ["__add__", "__sub__", "__mul__", "__truediv__",
                   "__mod__", "__pow__", "__and__", "__xor__", "__or__",
                   "dot", "mm"]:
  setattr(EntityTensor, _operation, _gen_placeholder_arithmetic(_operation))

class ConnectedTensor(object):
  def __init__(self, entity_tensor, connection_structures):
    self.entity_tensor = entity_tensor
    self.connection_structures = connection_structures

  def with_structure(self, connection_structure):
    result = copy(self)
    if isinstance(connection_structure, ConnectionStructure):
      result.connection_structures = [connection_structure]
    elif isinstance(connection_structure, list):
      result.connection_structures = connection_structure
    elif isinstance(connection_structure, int):
      result.connection_structures = self.connection_structures[connection_structure]
    else:
      result.connection_structures = [connection_structure(result.entity_tensor)]
    return result

  def new_like(self):
    result = copy(self)
    return result

  def __getitem__(self, idx):
    result = copy(self)
    result.entity_tensor = result.entity_tensor[idx]
    result.connection_structures = list(map(lambda x: x[idx], result.connection_structures))
    return result

  def __setitem__(self, idx, value):
    self.entity_tensor[idx] = value.entity_tensor

def _gen_placeholder_arithmetic_connected(operation):
  def _placeholder_arithmetic(self, other):
    result = copy(self)
    result.entity_tensor = getattr(self.entity_tensor, operation)(other.entity_tensor)
    return result
  return _placeholder_arithmetic

for _operation in ["__add__", "__sub__", "__mul__", "__truediv__",
                   "__mod__", "__pow__", "__and__", "__xor__", "__or__",
                   "dot", "mm"]:
  setattr(ConnectedTensor, _operation, _gen_placeholder_arithmetic_connected(_operation))
