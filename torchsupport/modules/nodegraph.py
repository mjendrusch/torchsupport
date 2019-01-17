import torch
import torch.nn as nn
import torch.nn.functional as func
import networkx as nx
from copy import copy, deepcopy

class NodeGraphTensor(object):
  def __init__(self, graphdesc=None):
    """Node-only graph tensor.

    Args:
      graphdesc (dict): dictionary of graph parameters.
    """
    self.recompute_adjacency_matrix = True
    self.adjacency_matrix = None
    self.recompute_laplacian = True
    self.laplacian = None
    self.is_subgraph = False
    self.offset = 0

    if graphdesc == None:
      self.num_graphs = 1
      self.graph_nodes = [0]
      self._adjacency = []

      self._node_tensor = torch.tensor([])
    else:
      self.num_graphs = graphdesc["num_graphs"]
      self.graph_nodes = graphdesc["graph_nodes"]
      self._adjacency = graphdesc["adjacency"]

      self._node_tensor = graphdesc["node_tensor"]

  def __repr__(self):
    class_name = self.__class__.__name__
    graphs = self.num_graphs
    nodes = len(self._node_tensor)
    edges = sum([len(edge) for edge in self._adjacency]) // 2
    adj = self.adjacency_matrix != None
    lap = self.laplacian != None
    return f"{class_name}({graphs}, {nodes}, {edges}, has_adjacency={adj}, has_laplacian={lap})" 

  @staticmethod
  def from_networkx(nx_graph, features=[]):
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
      for key in features:
        feats.append(nx_graph.nodes[node][key])
      if len(features) == 0:
        feats = [1.0]
      feats = torch.Tensor(feats)
      out.add_node(feats)
    for source, target in nx_graph.edges:
      out.add_edge(source, target)
    return out

  def to_networkx(self):
    """Creates a networkx graph from a `NodeGraphTensor`."""
    out = nx.Graph()
    for node, edges in enumerate(self._adjacency):
      out.add_node(node, features=self._node_tensor[node].numpy())
      for edge in edges:
        out.add_edge(node, edge)
    return out

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
    rng = self.graph_range(idx)
    return slice(rng.start, rng.stop)

  def laplacian_element(self, i, j):
    """Laplacian element.

    Args:
      i, j (int): position for element calculation.

    Returns:
      Element of the graph Laplacian at the subscript `i, j`.
    """
    if i == j:
      return len(self._adjacency[i])
    else:
      return -int(j in self._adjacency[i])

  def compute_laplacian(self):
    """Precomputes the graph Laplacian."""
    self.recompute_laplacian = False
    indices = [
      (node, edge)
      for node, edges in enumerate(self._adjacency)
      for edge in edges + [node]
    ]
    values = torch.zeros(len(indices))
    for idx, index in enumerate(indices):
      values[idx] = self.laplacian_element(*index)
    indices = torch.Tensor(indices).t()
    self.laplacian = torch.sparse_coo_tensor(
      indices, values,
      (len(self._adjacency), len(self._adjacency))
    )

  def decompute_laplacian(self):
    """Decomputes the graph Laplacian."""
    self.laplacian = None
    self.recompute_laplacian = True

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
      for node, edges in enumerate(self._adjacency):
        out[node] = len(edges) * vector[node] - vector[edges].sum(dim=0)
        if normalized:
          out[node] /= len(edges)
      return out
    else:
      if self.recompute_laplacian:
        self.compute_laplacian()
      out = self.laplacian.mm(vector)
      if normalized:
        norm = torch.Tensor([len(edges) for edges in self._adjacency])
        out /= norm
      return out

  def compute_adjacency_matrix(self):
    """Computes the graph adjacency matrix."""
    self.recompute_adjacency_matrix = False
    indices = torch.Tensor([
      (node, edge)
      for node, edges in enumerate(self._adjacency)
      for edge in edges + [node]
    ]).t()
    values = torch.ones(indices.size(1))
    self.adjacency_matrix = torch.sparse_coo_tensor(
      indices, values,
      (len(self._adjacency), len(self._adjacency))
    )

  def decompute_adjacency_matrix(self):
    """Decomputes the graph adjacency matrix."""
    self.recompute_adjacency_matrix = True
    self.adjacency_matrix = None

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
      for node, edges in enumerate(self._adjacency):
        out[node] = vector[edges].sum(dim=0)
      return out
    else:
      if self.recompute_adjacency_matrix:
        self.compute_adjacency_matrix()
      return self.adjacency_matrix(vector)

  def new_like(self):
    """Creates a new empty `NodeGraphTensor` with the same
    connectivity as `self`."""
    result = NodeGraphTensor()
    result.is_subgraph = self.is_subgraph
    result.offset = self.offset
    result.num_graphs = self.num_graphs
    result.graph_nodes = deepcopy(self.graph_nodes)
    result._adjacency = deepcopy(self._adjacency)
    result.decompute_laplacian()
    result.decompute_adjacency_matrix()
    return result

  def clone(self):
    """Clones a `NodeGraphTensor`."""
    result = self.new_like()
    result._node_tensor = self._node_tensor.clone()
    return result

  @property
  def adjacency(self):
    if self.is_subgraph:
      start = 0 if self.offset == 0 else self.nodes_including(self.offset - 1)
      stop = self.nodes_including(self.offset)
      return self._adjacency[start:stop]
    else:
      return self._adjacency

  @property
  def node_tensor(self):
    if self.is_subgraph:
      start = 0 if self.offset == 0 else self.nodes_including(self.offset - 1)
      stop = self.nodes_including(self.offset)
      return self._node_tensor[start:stop]
    else:
      return self._node_tensor

  @node_tensor.setter
  def node_tensor(self, value):
    if self.is_subgraph:
      start = 0 if self.offset == 0 else self.nodes_including(self.offset - 1)
      stop = self.nodes_including(self.offset)
      self._node_tensor[start:stop] = value
    else:
      self._node_tensor = value

  def nodes_including(self, graph_index):
    return sum(self.graph_nodes[:graph_index+1])

  def add_node(self, node_tensor):
    """Adds a node to the graph.
    
    Args:
      node_tensor (Tensor): tensor of node attributes to be added.

    Note:
      The graph Laplacian and adjacency matrix need to be recomputed
      afterwards, if used.
    """
    self.decompute_adjacency_matrix()
    self.decompute_laplacian()

    assert (self.num_graphs == 1)
    self.graph_nodes[self.offset] += 1
    self._adjacency.append([])
    if self._node_tensor.size(0) == 0:
      self._node_tensor = node_tensor.unsqueeze(0).unsqueeze(0)
    else:
      self._node_tensor = torch.cat(
        (self._node_tensor[:self.nodes_including(self.offset)],
        node_tensor.unsqueeze(0).unsqueeze(0),
        self._node_tensor[self.nodes_including(self.offset):]), 0)
    return self._node_tensor.size(0) - 1

  def add_edge(self, source, target):
    """Adds an edge to the graph.

    Args:
      source, target (int): the source and target nodes of the edge.

    Note:
      The graph Laplacian and adjacency matrix need to be recomputed
      afterwards, if used.
    """
    self.decompute_adjacency_matrix()
    self.decompute_laplacian()

    self._adjacency[source].append(target)
    self._adjacency[target].append(source)
    return len(self._adjacency[source]) - 1

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
    self.decompute_adjacency_matrix()
    self.decompute_laplacian()

    nodes_to_keep = []
    new_adjacency = []
    nodes = sorted(nodes)
    in_graph = 0
    node_sum = self.nodes_including(in_graph)
    nodes_removed = 0
    removed_index = 0
    for node in range(len(self._adjacency)):
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
    for node in range(len(self._adjacency)):
      new_adjacency.append([
        nodes_to_keep.index(target)
        for target in self._adjacency[node]
        if target in nodes_to_keep
      ])
    
    self._node_tensor = self._node_tensor[nodes_to_keep]
    self._adjacency = new_adjacency

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
    self.decompute_adjacency_matrix()
    self.decompute_laplacian()

    self._adjacency[source] = [x for x in self._adjacency[source] if x != target]
    self._adjacency[target] = [x for x in self._adjacency[target] if x != source]

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
    assert (self.num_graphs > 1)
    assert isinstance(idx, int) or isinstance(idx, slice)

    out = self.new_like()
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
    out._node_tensor = self._node_tensor[out_range.start:out_range.stop]
    out._adjacency = self._adjacency[out_range.start:out_range.stop]
    return out

  def append(self, graph_tensor):
    """Appends a `NodeGraphTensor` to the end of an existing `NodeGraphTensor`.
    
    Args:
      graph_tensor (NodeGraphTensor): tensor to be appended.
    """
    self.decompute_adjacency_matrix()
    self.decompute_laplacian()

    assert(self.offset == 0)
    self.num_graphs += graph_tensor.num_graphs
    self.adjacency += list(map(
      lambda x: x + len(self._adjacency), graph_tensor.adjacency))
    self.graph_nodes += graph_tensor.graph_nodes
    self._node_tensor = torch.cat((self.node_tensor, graph_tensor.node_tensor), 0)

# generate arithmetic ops on NodeGraphTensors
def _gen_placeholder_arithmetic(op):
  def _placeholder_arithmetic(self, other):
    assert other._node_tensor.size() == self._node_tensor.size()
    out = self.new_like()
    out._node_tensor = getattr(self._node_tensor, op)(other._node_tensor)
    return out
  return _placeholder_arithmetic

for op in ["__add__", "__sub__", "__mul__", "__truediv__",
           "__mod__", "__pow__", "__and__", "__xor__", "__or__",
           "dot", "mm"]:
  setattr(NodeGraphTensor, op, _gen_placeholder_arithmetic(op))

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
  else:
    out = graphs[0].new_like()
    node_tensors = [graph._node_tensor for graph in graphs]
    out._node_tensor = torch.cat(node_tensors, dim=dim)
    return out

class PartitionedNodeGraphTensor(NodeGraphTensor):
  def __init__(self, graphdesc=None):
    """Node-only graph tensor with multiple node types. See `NodeGraphTensor`."""
    super(PartitionedNodeGraphTensor, self).__init__(graphdesc=graphdesc)
    self.partition_view = None
    if graphdesc == None:
      self.partition = { None: [] }
    else:
      self.partition = graphdesc["partition"]

  def new_like(self):
    result = super(PartitionedNodeGraphTensor, self).new_like()
    result.partition_view = self.partition_view
    result.partition = self.partition
    return result

  def none(self):
    view = copy(self)
    view.partition_view = 'none'
    return view

  def all(self):
    view = copy(self)
    view.partition_view = None
    return view
  
  def add_kind(self, name):
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
      for node in range(len(self._adjacency))
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
        lambda x: x + len(self._adjacency), graph_tensor.partition[kind]))
    super(self, PartitionedNodeGraphTensor).append(graph_tensor)

def _batch_graphs(graphs):
  """Concatenates a list of graphs along the batch dimension.

  Args:
    graphs (iterable): graphs to be concatenated.
  """
  result = graphs[0].clone()
  for idx in range(1, len(graphs)):
    result.append(graphs[idx])
  return result

class ReNode(nn.Module):
  def __init__(self, reduction):
    """Turns all edges into nodes and all nodes into edges.

    Args:
      reduction (callable): reduction function merging two node
        features into a single edge feature.
    """
    super(ReNode, self).__init__()
    self.reduction = reduction
  
  def forward(self, graph):
    result = type(graph)()
    edges = (
      (node, edge)
      for node, edges in enumerate(graph._adjacency)
      for edge in edges
    )
    lookup = [
      []
      for node, edges in enumerate(graph._adjacency)
    ]
    new_nodes = []
    for idx, (source, target) in enumerate(edges):
      lookup[source].append(idx)
      lookup[target].append(idx)
      new_nodes.append(self.reduction(source, target))
    new_adjacency = [
      [
        neighbour
        for neighbour in lookup[source] + lookup[target]
        if neighbour != idx
      ]
      for idx, (source, target) in enumerate(edges)
    ]
    new_graph_nodes = [
      sum(map(len, graph._adjacency[graph.graph_slice(idx)]))
      for idx in range(graph.num_graphs)
    ]
    new_nodes = torch.cat(new_nodes, dim=0)
    result.num_graphs = graph.num_graphs
    result.graph_nodes = new_graph_nodes
    result._adjacency = new_adjacency
    result._node_tensor = new_nodes
    return result

class GCN(nn.Module):
  def __init__(self, in_channels, out_channels, augment=1, activation=nn.ReLU(), matrix_free=False):
    """Basic GCN.

    Args:
      in_channels, out_channels (int): number of in- and output features.
      augment (int): number of adjacency matrix augmentations.
      activation (callable): activation function.
      matrix_free (bool): use matrix-free matrix-vector multiplication?
    """
    super(GCN, self).__init__()
    self.linear = nn.Linear(in_channels, out_channels, bias=False)
    self.activation = activation
    self.matrix_free = matrix_free
    self.augment = augment

  def forward(self, graph):
    out = graph.new_like()
    new_nodes = self.linear(graph._node_tensor)
    normalization = torch.Tensor([len(edges) + 1 for edges in self._adjacency], dtype="float")
    for idx in range(self.augment):
      new_nodes = (self.adjacency_action(new_nodes, matrix_free=self.matrix_free) + new_nodes)
      new_nodes /= normalization
    out._node_tensor = self.activation(new_nodes)
    return out

class MultiscaleGCN(nn.Module):
  def __init__(self, in_channels, out_channels, scales, activation=nn.ReLU, matrix_free=False):
    """Multiscale GCN.

    Args:
      in_channels, out_channels (int): number of in- and output features.
      scales (list int): number of adjacency matrix augmentations.
      activation (callable): activation function.
      matrix_free (bool): use matrix-free matrix-vector multiplication?
    """
    super(MultiscaleGCN, self).__init__()
    self.modules = nn.ModuleList([
      GCN(in_channels, out_channels, augment=scale,
              activation=activation, matrix_free=matrix_free)
      for scale in scales
    ])

  def forward(self, graph):
    outputs = []
    for module in self.modules
      outputs.append(module(graph))
    return cat(outputs, dim=1)

class ChebyshevConv(nn.Module):
  def __init__(self, in_channels, out_channels, depth=2,
               activation=nn.ReLU(), matrix_free=False):
    """Chebyshev polynomial-based approximate spectral graph convolution.
    
    Args:
      in_channels, out_channels (int): number of input and output features.
      depth (int): depth of the Chebyshev approximation.
      activation (callable): activation function.
      matrix_free (bool): do not materialize Laplacian explicitly?
    """
    super(ChebyshevConv, self).__init__()
    self.linear = nn.Linear(in_channels, out_channels, bias=False)
    self.depth = depth
    self.activation = activation
    self.matrix_free = matrix_free

  def forward(self, graph):
    m2_nodes = self.linear(graph._node_tensor)
    m1_nodes = graph.laplacian_action(m2_nodes)
    out_nodes = m1_nodes + m2_nodes
    for idx in range(2, self.depth):
      m2_nodes = m1_nodes
      m1_nodes += 2 * graph.laplacian_action(
        m1_nodes, matrix_free=self.matrix_free, normalized=True
      ) - m2_nodes
      out_nodes += m1_nodes
    out = graph.new_like()
    out._node_tensor = self.activation(out_nodes)
    return out

class ARMAConv(nn.Module):
  def __init__(self, in_channels, out_channels, share=True,
               width=2, depth=2, activation=nn.ReLU(),
               matrix_free=False):
    """Auto-regressive moving average-based approximate spectral graph convolution.
    
    Args:
      in_channels, out_channels (int): number of input and output features.
      width (int): width of the ARMA filter stack.
      depth (int): depth of the ARMA approximation.
      activation (callable): activation function.
      matrix_free (bool): do not materialize Laplacian explicitly?
    """
    super(ARMAConv, self).__init__()
    self.width = width
    self.depth = depth
    self.activation = activation
    self.matrix_free = matrix_free
    self.share = share

    # width × different linear ops
    self.preprocess = nn.Linear(in_channels, out_channels * width)
    self.propagate = nn.Conv1d(out_channels * width, out_channels * width, 1, groups=width)
    self.merge = nn.Linear(in_channels, out_channels * width)

  def forward(self, graph):
    nodes = graph._node_tensor

    out = self.preprocess(nodes)
    out = out.reshape(out.size(0), out.size(1) * out.size(2), 1)
    out += self.merge(nodes).reshape(out.size(0), out.size(1) * out.size(2), 1)
    out = self.activation(out)
    for idx in range(self.depth - 1):
      out -= graph.laplacian_action(out)
      out = self.propagate(out)
      out += self.merge(nodes).reshape(out.size(0), out.size(1) * out.size(2), 1)
      out = self.activation(out)

    out = out.reshape(nodes.size(0), nodes.size(1), self.width)
    out = func.adaptive_avg_pool1d(out, 1).reshape(
      nodes.size(0), -1
    ).unsqueeze(2)
    result = graph.new_like()
    result._node_tensor = out
    return result

class AllNodes(nn.Module):
  def __init__(self, node_update):
    """Applies a node update function to all nodes in a graph.

    Args:
      node_update (nn.Module): update to apply to all nodes.
    """
    super(AllNodes, self).__init__()
    self.node_update = node_update

  def forward(self, graph):
    out = graph.new_like()
    if isinstance(out, PartitionedNodeGraphTensor) and out.partition_view != None:
      out._node_tensor = graph._node_tensor.clone()
      partition = out.partition[out.partition_view]
      out._node_tensor[partition, :] = self.node_update(out._node_tensor[partition, :])
    else:
      out._node_tensor = self.node_update(out._node_tensor)
    return out

def LinearOnNodes(insize, outsize):
  """Applies a linear module at each node in a graph.
  
  Args:
    insize (int): number of input features.
    outsize (int): number of output features.

  Returns:
    `AllNodes` wrapping a `nn.Linear` module.
  """
  lin = nn.Linear(insize, outsize)
  def mod(x):
    x = x.view(x.size()[0], -1)
    x = lin(x)
    x = x.unsqueeze(1)
    return x
  return AllNodes(mod)

def StandardNodeTraversal(depth):
  def function(graph, entity, d=depth):
    if d == 0:
        return [entity]
    else:
      nodes = [entity]
      edges = graph.adjacency[entity]
      nodes += edges
      for new_node in edges:
        if new_node != entity:
          new_nodes = function(
            graph, new_node, d - 1
          )
          nodes += new_nodes
      nodes = list(set(nodes))
      return nodes
  return function

class NodeGraphNeighbourhood(nn.Module):
  def __init__(self, reducer, traversal=StandardNodeTraversal(1), order=None):
    """Applies a reduction function to the neighbourhood of each node.

    Args:
      reducer (callable): reduction function.
      traversal (callable): node traversal for generating node neighbourhoods.
      order (callable): optional sorting function for sorting node neighbourhoods.
    """
    super(NodeGraphNeighbourhood, self).__init__()
    self.reducer = reducer
    self.traversal = traversal
    self.order = order

  def forward(self, graph, include_self=True):
    full_nodes = []
    partition = range(graph._node_tensor.size(0))
    if isinstance(graph, PartitionedNodeGraphTensor) and graph.partition_view != None:
      partition = graph.partition[graph.partition_view]
    for node in partition:
      nodes = self.traversal(graph, node)
      if self.order != None:
        nodes = self.order(graph, nodes)
      full_nodes.append(nodes)
    reduced_nodes = self.reducer(graph, full_nodes)
    out = graph.new_like()
    out._node_tensor = torch.cat((graph._node_tensor, reduced_nodes), 1)
    return out

def _node_neighbourhood_assignment(linears, source, target):
  def reducer(graph, nodes):
    source_tensor = source(graph._node_tensor)
    target_tensor = target(graph._node_tensor)
    weight_tensors = []
    for module in linears:
      weight_tensors.append(module(graph._node_tensor).unsqueeze(0))
    weighted = torch.cat(weight_tensors, dim=0)
    new_node_tensor = torch.zeros_like(graph._node_tensor)
    for idx, node in enumerate(nodes):
      assignment_tensor = func.softmax(source_tensor[idx] + target_tensor[node]).unsqueeze(0)
      new_node_tensor[idx] = (assignment_tensor * weighted[node]).mean(dim=0).squeeze(0)
    return new_node_tensor
  return reducer

def NodeNeighbourhoodAssignment(in_channels, out_channels, size,
                                traversal=StandardNodeTraversal(1)):
  """Aggregates a node neighbourhood using soft weight assignment. (FeaStNet)

  Args:
    in_channels (int): number of input features.
    out_channels (int): number of output features.
    size (int): number of distinct weight matrices (equivalent to kernel size).
    traversal (callable): node traversal for generating node neighbourhoods.
  """
  linears = nn.ModuleList([
    nn.Linear(in_channels, out_channels)
    for _ in range(size)
  ])
  source = nn.Linear(in_channels, 1)
  target = nn.Linear(in_channels, 1)
  return NodeGraphNeighbourhood(
    _node_neighbourhood_assignment(linears, source, target),
    traversal=traversal
  )

def _node_neighbourhood_attention(att):
  def reducer(graph, nodes):
    new_node_tensor = torch.zeros_like(graph._node_tensor)
    for idx, node in enumerate(nodes):
      local_tensor = graph._node_tensor[node]
      attention = att(torch.cat((local_tensor, graph._node_tensor[idx])))
      reduced = torch.Tensor.sum(attention * local_tensor, dim=1)
      new_node_tensor[idx] = reduced if isinstance(reduced, torch.Tensor) else reduced[0]
    return new_node_tensor
  return reducer

def NodeNeighbourhoodAttention(attention, traversal=StandardNodeTraversal(1)):
  """Aggregates a node neighbourhood using an attention mechanism.
  
  Args:
    attention (callable): attention mechanism to be used.
    traversal (callable): node traversal for generating node neighbourhoods.
  """
  return NodeGraphNeighbourhood(
    _node_neighbourhood_attention(attention),
    traversal=traversal
  )

def neighbourhood_to_adjacency(neighbourhood):
  size = torch.Size([len(neighbourhood), len(neighbourhood)])
  indices = []
  for idx, nodes in enumerate(neighbourhood):
    for node in nodes:
      indices.append([idx, node])
      indices.append([node, idx])
  indices = torch.Tensor(list(set(indices)))
  values = torch.ones(indices.size(0))
  return torch.sparse_coo_tensor(indices, values, size)

def _node_neighbourhood_sparse_attention(embedding, att, att_p):
  def reducer(graph, nodes):
    embedding = embedding(graph._node_tensor)
    local_attention = att.dot(embedding)
    neighbour_attention = att_p.dot(embedding)
    adjacency = neighbourhood_to_adjacency(nodes)
    new_node_tensor = local_attention + torch.mm(adjacency, neighbour_attention)
    return new_node_tensor
  return reducer

def NodeNeighbourhoodSparseAttention(size, traversal=StandardNodeTraversal(1)):
  """Aggregates a node neighbourhood using a sparse attention mechanism.

  Args:
    size (int): size of the attention embedding.
    traversal (callable): node traversal for generating node neighbourhoods.
  """
  embedding = nn.Linear(size, size)
  att = torch.randn(size, requires_grad=True)
  att_p = torch.randn(size, requires_grad=True)
  return NodeGraphNeighbourhood(
    _node_neighbourhood_sparse_attention(embedding, att, att_p),
    traversal=traversal
  )

def _node_neighbourhood_dot_attention(embedding, att, att_p):
  def reducer(graph, nodes):
    embedding = embedding(graph._node_tensor)
    local_attention = att.dot(embedding)
    neighbour_attention = att_p.dot(embedding)
    for idx, node in enumerate(nodes):
      reduced = torch.Tensor.sum((local_attention[idx] + neighbour_attention[node]) * graph.node_tensor[node], dim=1)
      new_node_tensor[idx] = reduced if isinstance(reduced, torch.Tensor) else reduced[0]
    return new_node_tensor
  return reducer

def NodeNeighbourhoodDotAttention(size, traversal=StandardNodeTraversal(1)):
  """Aggregates a node neighbourhood using a pairwise dot-product attention mechanism.
  
  Args:
    size (int): size of the attention embedding.
    traversal (callable): node traversal for generating node neighbourhoods.
  """
  embedding = nn.Linear(size, size)
  att = torch.randn(size, requires_grad=True)
  att_p = torch.randn(size, requires_grad=True)
  return NodeGraphNeighbourhood(
    _node_neighbourhood_dot_attention(embedding, att, att_p),
    traversal=traversal
  )

def _node_neighbourhood_reducer(red):
  def reducer(graph, nodes):
    new_node_tensor = torch.zeros_like(graph.node_tensor)
    for idx, node in enumerate(nodes):
      reduced = red(graph.node_tensor[node], dim=1)
      new_node_tensor[idx] = reduced if isinstance(reduced, torch.Tensor) else reduced[0]
    return new_node_tensor
  return reducer

def NodeNeighbourhoodMean(traversal=StandardNodeTraversal(1)):
  """Aggregates a node neighbourhood using the mean of neighbour features."""
  return NodeGraphNeighbourhood(
    _node_neighbourhood_reducer(torch.Tensor.mean),
    traversal=traversal
  )

def NodeNeighbourhoodSum(traversal=StandardNodeTraversal(1)):
  """Aggregates a node neighbourhood using the sum of neighbour features."""
  return NodeGraphNeighbourhood(
    _node_neighbourhood_reducer(torch.Tensor.sum),
    traversal=traversal
  )

def NodeNeighbourhoodMax(traversal=StandardNodeTraversal(1)):
  """Aggregates a node neighbourhood using the maximum of neighbour features."""
  return NodeGraphNeighbourhood(
    _node_neighbourhood_reducer(torch.Tensor.max),
    traversal=traversal
  )

def NodeNeighbourhoodMin(traversal=StandardNodeTraversal(1)):
  """Aggregates a node neighbourhood using the minimum of neighbour features."""
  return NodeGraphNeighbourhood(
    _node_neighbourhood_reducer(torch.Tensor.min),
    traversal=traversal
  )

class GraphResBlock(nn.Module):
  def __init__(self, channels, aggregate=NodeNeighbourhoodMax,
               activation=nn.ReLU()):
    """Residual block for graph networks.

    Args:
      channels (int): number of input and output features.
      aggregate (nn.Module): neighbourhood aggregation function.
      activation (nn.Module): activation function. Defaults to ReLU.
    """
    self.activation = activation
    self.aggregate = aggregate
    self.linear = LinearOnNodes(2 * channels, channels)

  def forward(self, input):
    out = self.aggregate(input)
    out = self.linear(out)
    out = self.activation(out + input)
    return out

class LearnedColorPool(nn.Module):
  def __init__(self, channels, pooling, order=None,
               traversal=StandardNodeTraversal(1),
               attention_activation=nn.Tanh(),
               activation=nn.ReLU()):
    """Generalization of pooling from images to graphs, using learned
    pooling centers and attention.

    Args:
      channels (int): number of node features.
      pooling (callable): pooling function.
      order (callable): function specifying a sort order for nodes to be pooled.
      traversal (callable): function computing a neighbourhood traversal for a given node.
    """
    super(LearnedColorPool, self).__init__()
    self.embedding = nn.Linear(channels, channels)
    self.attention_activation = attention_activation
    self.activation = activation
    self.chosen = None
    self.color_pool = ColorPool(
      pooling, order=order, coloring=lambda x: self.chosen, traversal=traversal
    )

  def forward(self, graph):
    embedding = self.embedding(graph)
    attention = embedding.dot(graph)
    all_topk = []
    graph_sum = 0
    for idx, graph_nodes in enumerate(graph.graph_nodes):
      topk, indices = torch.topk(
        attention._node_tensor[graph.graph_range(idx)],
        graph_nodes // 2
      )
      all_topk.append(indices + graph_sum)
      graph_sum += graph_nodes
    all_topk = torch.cat(all_topk, dim=0).reshape(-1)
    attention._node_tensor = self.attention_activation(attention._node_tensor)
    attended = self.activation(graph * abs(attention) + graph)
    self.chosen = all_topk
    return self.color_pool(attended)

def MinimumDegreeNodeColoring():
  """Partitions a graph using a heuristic choosing all nodes with non-minimum
  connectivity in their neighbourhood.
  """
  def color(graph):
    chosen = []
    for idx in range(len(graph._adjacency)):
      lengths = [len(graph._adjacency[edge]) for edge in graph._adjacency[idx]]
      self_length = len(graph._adjacency[idx])
      minimum = min(lengths + [self_length])
      if self_length != minimum:
        chosen.append(idx)
    return torch.LongTensor(chosen)
  return color

def MaximumEigenvectorNodeColoring(n_iter=2, matrix_free=True):
  """Partitions a graph using its Laplacian's largest eigenvector.

  Args:
    n_iter (int): number of power iterations for eigenvector estimate.
    matrix_free (bool): construct Laplacian elements on the fly?

  Returns:
    graph partition obtained by choosing all nodes with values > 0
    in an approximation to the largest-eigenvalue eigenvector of the
    graph Laplacian, obtained by power iteration.

  Note:
    For large graphs and batch sizes, this may result in excessive
    memory consumption if not using `matrix_free = True`. On the
    other hand, setting `matrix_free = True` results in higher
    computational load.
  """
  def color(graph):
    values = func.normalize(torch.randn(graph._node_tensor.size(0), 1), dim=0)
    for idx in range(n_iter):
      values = func.normalize(graph.laplacian_action(values, matrix_free=matrix_free), dim=0)
    chosen = (values > 0).reshape(-1).nonzero().numpy()
    return torch.LongTensor(chosen)
  return color

class ColorPool(nn.Module):
  def __init__(self, pooling, order=None,
               coloring=MaximumEigenvectorNodeColoring(),
               traversal=StandardNodeTraversal(1)):
    """Generalization of (maximum-) pooling from images to graphs.
    Chooses a set of pooling centers using a user specified `coloring`,
    and pools the pooling centers' neighbourhoods generated using a user
    specified `traversal` according to a `pooling` function.
    
    Args:
      pooling (callable): pooling function.
      order (callable): function specifying a sort order for nodes to be pooled.
      coloring (callable): function specifying pooling centers on a graph.
      traversal (callable): function computing a neighbourhood traversal for a given node.
    """
    super(ColorPool, self).__init__()
    self.pooling = pooling
    self.order = order
    self.coloring = coloring
    self.traversal = traversal

  def _pooled_nodes(self, selected_nodes, neighbourhoods):
    return [
      node
      for node in set(sum(neighbourhoods, []))
      if node not in selected_nodes
    ]

  def _expanded_edges(self, graph, nodes_to_delete):
    return [
      (node, target)
      for skip in nodes_to_delete
      for idx, node in enumerate(graph._adjacency[skip])
      for target in graph._adjacency[skip][idx+1:]
      if (node not in nodes_to_delete) and (target not in nodes_to_delete)
    ]

  def _ordered_neighbourhoods(self, graph, selected_nodes):
    neighbourhoods = []
    for node in selected_nodes:
      nodes = self.traversal(graph, node)
      if self.order != None:
        nodes = self.order(graph, nodes)
      neighbourhoods.append(nodes)
    return neighbourhoods

  def forward(self, graph):
    # select pooling centers and compute pooling neighbourhoods:
    pooling_centers = self.coloring(graph)
    neighbourhoods = self._ordered_neighbourhoods(graph, pooling_centers)

    # nodes and edges to be amended:
    nodes_to_delete = self._pooled_nodes(pooling_centers, neighbourhoods)
    edges_to_add = self._expanded_edges(graph, nodes_to_delete)

    # pool and cull graph:
    out = graph.clone()
    out.add_edges(edges_to_add)
    out.delete_nodes(nodes_to_delete)
    out._node_tensor = torch.cat([
      self.pooling(graph._node_tensor[[node] + neighbourhoods[idx]])
      for idx, node in enumerate(pooling_centers)
    ], dim=0)
    return out, pooling_centers

class ColorUnpool(nn.Module):
  def __init__(self, unpool):
    """Generalization of (maximum-) unpooling from images to graphs.
    Unpools a graph previously pooled using `ColorPool` by applying
    a partial inverse unpool operation broadcasting data from pooling
    centers to pooled nodes.
    
    Args:
      unpool (callable): unpooling operation.
    """
    self.unpool = unpool

  def forward(self, input, indices, guide_graph):
    out = guide_graph.new_like()
    out._node_tensor = torch.zeros((guide_graph.size(0), *input.size[1:]))
    node_sum = 0
    for node, edges in enumerate(guide_graph._adjacency):
      if node in indices:
        out._node_tensor[node] = input._node_tensor[indices.index(node)]
        for target in edges:
          out._node_tensor[target] = self.unpool(
            out._node_tensor[target],
            input._node_tensor[indices.index(node)]
          )
    return out
