from copy import copy, deepcopy

import torch
import networkx as nx

class Incoming(object):
  def __init__(self, data):
    self.data = data

class Outgoing(object):
  def __init__(self, data):
    self.data = data

class NodeGraphTensor(object):
  """Node-only graph tensor."""

  def __init__(self, graphdesc=None):
    """Node-only graph tensor.

    Args:
      graphdesc (dict): dictionary of graph parameters.
    """
    self._recompute_adjacency_matrix = True
    self._adjacency_matrix = None
    self._recompute_laplacian = True
    self._laplacian = None
    self.offset = 0
    self.directed = False

    if graphdesc is None:
      self.num_graphs = 1
      self.graph_nodes = [0]
      self.adjacency = []

      self.node_tensor = torch.Tensor([])
    else:
      self.num_graphs = graphdesc["num_graphs"]
      self.graph_nodes = graphdesc["graph_nodes"]
      self.adjacency = graphdesc["adjacency"]

      self.node_tensor = graphdesc["node_tensor"]

  def __repr__(self):
    class_name = self.__class__.__name__
    graphs = self.num_graphs
    nodes = len(self.node_tensor)
    edges = sum([len(edge) for edge in self.adjacency]) // 2
    adj = self._adjacency_matrix is not None
    lap = self._laplacian is not None
    return f"{class_name}({graphs}, {nodes}, {edges}, has_adjacency={adj}, has_laplacian={lap})"

  @staticmethod
  def from_networkx(nx_graph, features=None):
    """Creates a new `NodeGraphTensor` from an `nx.Graph`.

    Args:
      nx_graph (nx.Graph): networkx graph input.
      features (list str): list of feature keys.

    Returns:
      `NodeGraphTensor` containing the data from the input `nx.Graph`.
    """
    out = NodeGraphTensor()
    for node in nx_graph.nodes:
      feats = []
      if not features:
        feats = [1.0]
      else:
        for key in features:
          feats.append(nx_graph.nodes[node][key])
      feats = torch.Tensor(feats)
      out.add_node(feats)
    for source, target in nx_graph.edges:
      out.add_edge(source, target)
    return out

  def to_networkx(self):
    """Creates a networkx graph from a `NodeGraphTensor`."""
    out = nx.Graph()
    for node, edges in enumerate(self.adjacency):
      out.add_node(node, features=self.node_tensor[node].numpy())
      for edge in edges:
        out.add_edge(node, edge)
    return out

  def node_graph(self, node):
    """Finds the index of the graph a node belongs to.

    Args:
      node (int): node index.

    Returns:
      Index of the graph `node` belongs to.
    """
    assert node < self.node_tensor.size(0)
    total_nodes = 0
    for graph, graph_nodes in enumerate(self.graph_nodes):
      if total_nodes <= node < total_nodes + graph_nodes:
        return graph
      total_nodes += graph_nodes
    return -1

  def graph_range(self, idx):
    """Range of nodes contained in the `idx`th graph.

    Args:
      idx (int): graph whose node range is to be computed.

    Returns:
      Range of nodes in the `idx`th graph.
    """
    assert idx < self.num_graphs
    start = 0
    for graph in range(idx):
      start += self.graph_nodes[graph]
    stop = start + self.graph_nodes[idx]
    return range(start, stop)

  def graph_slice(self, idx):
    """Slice of nodes contained in the `idx`th graph. See `graph_range`."""
    rng = self.graph_range(idx)
    return slice(rng.start, rng.stop)

  def _laplacian_element(self, i, j):
    """Laplacian element.

    Args:
      i, j (int): position for element calculation.

    Returns:
      Element of the graph Laplacian at the subscript `i, j`.
    """
    if i == j:
      return len(self.adjacency[i])
    else:
      return -int(j in self.adjacency[i])

  def _compute_laplacian(self):
    """Precomputes the graph Laplacian."""
    self._recompute_laplacian = False
    indices = [
      (node, edge)
      for node, edges in enumerate(self.adjacency)
      for edge in edges + [node]
    ]
    values = torch.zeros(len(indices))
    for idx, index in enumerate(indices):
      values[idx] = self._laplacian_element(*index)
    indices = torch.Tensor(indices).t()
    self._laplacian = torch.sparse_coo_tensor(
      indices, values,
      (len(self.adjacency), len(self.adjacency))
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
      for node, edges in enumerate(self.adjacency):
        out[node] = len(edges) * vector[node] - vector[edges].sum(dim=0)
        if normalized:
          out[node] /= len(edges)
      return out
    if self._recompute_laplacian:
      self._compute_laplacian()
    out = self._laplacian.mm(vector)
    if normalized:
      norm = torch.Tensor([len(edges) for edges in self.adjacency])
      out /= norm
    return out

  def _compute_adjacency_matrix(self):
    """Computes the graph adjacency matrix."""
    self._recompute_adjacency_matrix = False
    indices = torch.Tensor([
      (node, edge)
      for node, edges in enumerate(self.adjacency)
      for edge in edges + [node]
    ]).t()
    values = torch.ones(indices.size(1))
    self._adjacency_matrix = torch.sparse_coo_tensor(
      indices, values,
      (len(self.adjacency), len(self.adjacency))
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
      for node, edges in enumerate(self.adjacency):
        out[node] = vector[edges].sum(dim=0)
      return out
    else:
      if self._recompute_adjacency_matrix:
        self._compute_adjacency_matrix()
      return self._adjacency_matrix(vector)

  def new_like(self):
    """Creates a new empty `NodeGraphTensor` with the same
    connectivity as `self`."""
    result = NodeGraphTensor()
    result.offset = self.offset
    result.num_graphs = self.num_graphs
    result.graph_nodes = deepcopy(self.graph_nodes)
    result.adjacency = deepcopy(self.adjacency)
    return result

  def clone(self):
    """Clones a `NodeGraphTensor`."""
    result = self.new_like()
    result.node_tensor = self.node_tensor.clone()
    return result

  def nodes_including(self, graph_index):
    """Computes the number of nodes in all graphs up to `graph_index`."""
    return sum(self.graph_nodes[:graph_index+1])

  def add_node(self, node_tensor):
    """Adds a node to the graph.

    Args:
      node_tensor (Tensor): tensor of node attributes to be added.

    Note:
      The graph Laplacian and adjacency matrix need to be recomputed
      afterwards, if used.
    """
    self._decompute_adjacency_matrix()
    self._decompute_laplacian()

    assert self.num_graphs == 1
    self.graph_nodes[self.offset] += 1
    self.adjacency.append([])
    if self.node_tensor.size(0) == 0:
      self.node_tensor = node_tensor.unsqueeze(0).unsqueeze(0)
    else:
      self.node_tensor = torch.cat(
        (self.node_tensor[:self.nodes_including(self.offset)],
         node_tensor.unsqueeze(0).unsqueeze(0),
         self.node_tensor[self.nodes_including(self.offset):]), 0)
    return self.node_tensor.size(0) - 1

  def add_edge(self, source, target):
    """Adds an edge to the graph.

    Args:
      source, target (int): the source and target nodes of the edge.

    Note:
      The graph Laplacian and adjacency matrix need to be recomputed
      afterwards, if used.
    """
    self._decompute_adjacency_matrix()
    self._decompute_laplacian()

    if self.directed:
      self.adjacency[source].append(Outgoing(target))
      self.adjacency[target].append(Incoming(source))
    else:
      self.adjacency[source].append(target)
      self.adjacency[target].append(source)
    return len(self.adjacency[source]) - 1

  def add_edges(self, edges):
    """Adds a list of edges to the graph.

    Args:
      edges (list (tuple int)): list of edges to add.

    Note:
      The graph Laplacian and adjacency matrix need to be recomputed
      afterwards, if used.
    """
    for edge in edges:
      self.add_edge(*edge)

  def delete_nodes(self, nodes):
    """Deletes a list of nodes from the graph.

    Args:
      nodes (list int): list of nodes to be deleted.

    Note:
      The graph Laplacian and adjacency matrix need to be recomputed
      afterwards, if used.
    """
    self._decompute_adjacency_matrix()
    self._decompute_laplacian()

    nodes_to_keep = []
    new_adjacency = []
    nodes = sorted(nodes)
    in_graph = 0
    node_sum = self.nodes_including(in_graph)
    nodes_removed = 0
    removed_index = 0
    for node in range(len(self.adjacency)):
      if nodes[removed_index] == node:
        while node > node_sum:
          self.graph_nodes[in_graph] -= nodes_removed
          in_graph += 1
          node_sum = self.nodes_including(in_graph)
          nodes_removed = 0
        nodes_removed += 1
        removed_index += 1
      else:
        nodes_to_keep.append(node)
    for node in range(len(self.adjacency)):
      new_adjacency.append([
        nodes_to_keep.index(target)
        for target in self.adjacency[node]
        if target in nodes_to_keep
      ])

    self.node_tensor = self.node_tensor[nodes_to_keep]
    self.adjacency = new_adjacency

  def delete_node(self, node):
    """Deletes a single node from the graph.

    Args:
      node (int): node to be deleted.

    Note:
      The graph Laplacian and adjacency matrix need to be recomputed
      afterwards, if used.
    """
    self.delete_nodes([node])

  def delete_edge(self, source, target):
    """Deletes a single edge from the graph.

    Args:
      source, target (int): source and target of the edge to be deleted.

    Note:
      The graph Laplacian and adjacency matrix need to be recomputed
      afterwards, if used.
    """
    self._decompute_adjacency_matrix()
    self._decompute_laplacian()

    self.adjacency[source] = [x for x in self.adjacency[source] if x != target]
    self.adjacency[target] = [x for x in self.adjacency[target] if x != source]

  def delete_edges(self, edges):
    """Deletes a list of edges from the graph.

    Args:
      edges (list (tuple int)): list of edges to be deleted.

    Note:
      The graph Laplacian and adjacency matrix need to be recomputed
      afterwards, if used.
    """
    for edge in edges:
      self.delete_edge(*edge)

  def __getitem__(self, idx):
    assert isinstance(idx, (int, slice, tuple))

    out = self.new_like()
    further_indices = None
    if isinstance(idx, tuple) and len(idx) > 1:
      further_indices = idx[1:]
      idx = idx[0]
    if isinstance(idx, int):
      out_range = self.graph_range(idx)
      out.num_graphs = 1
      out.graph_nodes = [self.graph_nodes[idx]]
    elif isinstance(idx, slice):
      out_range = slice(
        self.graph_range(idx.start).start,
        self.graph_range(idx.stop-1).stop
      )
      out.num_graphs = max(0, idx.stop - idx.start)
      out.graph_nodes = [
        self.graph_nodes[k]
        for k in range(idx.start, idx.stop)
      ]
    out.node_tensor = self.node_tensor[out_range.start:out_range.stop]
    if further_indices is not None:
      out.node_tensor = out.node_tensor[further_indices]
    out.adjacency = self.adjacency[out_range.start:out_range.stop]
    if further_indices is not None:
      out.adjacency = out.adjacency[further_indices[0]]
    return out

  def __setitem__(self, idx, value):
    assert isinstance(idx, (int, slice, tuple))

    further_indices = None
    if isinstance(idx, tuple) and len(idx) > 1:
      further_indices = idx[1:]
      idx = idx[0]
    if isinstance(idx, int):
      out_range = self.graph_range(idx)
    elif isinstance(idx, slice):
      out_range = slice(
        self.graph_range(idx.start).start,
        self.graph_range(idx.stop-1).stop
      )
    if further_indices is not None:
      self.node_tensor[out_range.start:out_range.stop, further_indices] = value.node_tensor
    else:
      self.node_tensor[out_range.start:out_range.stop] = value.node_tensor
    if further_indices is not None:
      self.adjacency[out_range.start:out_range.stop][further_indices[0]] = value.adjacency
    else:
      self.adjacency[out_range.start:out_range.stop] = value.adjacency

  def append(self, graph_tensor):
    """Appends a `NodeGraphTensor` to the end of an existing `NodeGraphTensor`.

    Args:
      graph_tensor (NodeGraphTensor): tensor to be appended.
    """
    self._decompute_adjacency_matrix()
    self._decompute_laplacian()

    assert self.offset == 0
    self.num_graphs += graph_tensor.num_graphs
    self.adjacency += list(map(
      lambda x: x + len(self.adjacency), graph_tensor.adjacency))
    self.graph_nodes += graph_tensor.graph_nodes
    self.node_tensor = torch.cat((self.node_tensor, graph_tensor.node_tensor), 0)

# generate arithmetic ops on NodeGraphTensors
def _gen_placeholder_arithmetic(operation):
  def _placeholder_arithmetic(self, other):
    assert other.node_tensor.size() == self.node_tensor.size()
    out = self.new_like()
    out.node_tensor = getattr(self.node_tensor, operation)(other.node_tensor)
    return out
  return _placeholder_arithmetic

for _operation in ["__add__", "__sub__", "__mul__", "__truediv__",
                   "__mod__", "__pow__", "__and__", "__xor__", "__or__",
                   "dot", "mm"]:
  setattr(NodeGraphTensor, _operation, _gen_placeholder_arithmetic(_operation))

def cat(graphs, dim=0):
  """Contatenates a list of `NodeGraphTensor`s into a single `NodeGraphTensor`.

  Args:
    graphs (iterable): iterable of `NodeGraphTensor`s to be concatentated.
    dim (int): dimension along which to concatenate the `NodeGraphTensor`s.

  Returns:
    A `NodeGraphTensor` containing the concatenation of the input graphs along
      the input dimension.
  """
  if dim == 0:
    return _batch_graphs(graphs)
  out = graphs[0].new_like()
  node_tensors = [graph.node_tensor for graph in graphs]
  out.node_tensor = torch.cat(node_tensors, dim=dim)
  return out

class PartitionedNodeGraphTensor(NodeGraphTensor):
  """Node-only graph tensor with multiple node types. See `NodeGraphTensor`."""

  def __init__(self, graphdesc=None):
    """Node-only graph tensor with multiple node types. See `NodeGraphTensor`."""
    super(PartitionedNodeGraphTensor, self).__init__(graphdesc=graphdesc)
    self.partition_view = None
    if graphdesc is None:
      self.partition = {None: []}
    else:
      self.partition = graphdesc["partition"]

  def new_like(self):
    result = super(PartitionedNodeGraphTensor, self).new_like()
    result.partition_view = self.partition_view
    result.partition = self.partition
    return result

  def none(self):
    """Returns the partition of nodes with no kind."""
    view = copy(self)
    view.partition_view = 'none'
    return view

  def all(self):
    """Returns the partition containing all nodes."""
    view = copy(self)
    view.partition_view = None
    return view

  def add_kind(self, name):
    """Adds a node kind to the graph.

    Args:
      name (str): name of the new node kind.
    """
    self.partition[name] = []
    def _function():
      view = copy(self)
      view.partition_view = name
      return view
    self.__dict__[name] = _function
    return self.partition[name]

  def add_node(self, node_tensor, kind=None):
    node = super(PartitionedNodeGraphTensor, self).add_node(node_tensor)
    self.partition[kind].append(node)
    return node

  def delete_nodes(self, nodes):
    nodes_to_keep = [
      node
      for node in range(len(self.adjacency))
      if node not in nodes
    ]
    self.partition = {
      kind : [
        nodes_to_keep.index(node)
        for node in self.partition[kind]
        if node not in nodes
      ]
      for kind in self.partition
    }
    super(PartitionedNodeGraphTensor, self).delete_nodes(nodes)

  def append(self, graph_tensor):
    for kind in self.partition:
      self.partition[kind] += list(map(
        lambda x: x + len(self.adjacency), graph_tensor.partition[kind]))
    super(PartitionedNodeGraphTensor, self).append(graph_tensor)

def _batch_graphs(graphs):
  """Concatenates a list of graphs along the batch dimension.

  Args:
    graphs (iterable): graphs to be concatenated.
  """
  result = graphs[0].clone()
  for idx in range(1, len(graphs)):
    result.append(graphs[idx])
  return result
