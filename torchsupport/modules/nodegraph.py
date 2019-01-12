import torch
import torch.nn as nn
import torch.nn.functional as func
from copy import copy, deepcopy

class NodeGraphTensor(object):
  def __init__(self, graphdesc=None):
    """Node-only graph tensor.

    Args:
      graphdesc (dict): dictionary of graph parameters.
    """
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

  def new_like(self):
    result = NodeGraphTensor()
    result.is_subgraph = self.is_subgraph
    result.offset = self.offset
    result.num_graphs = self.num_graphs
    result.graph_nodes = deepcopy(self.graph_nodes)
    result._adjacency = deepcopy(self._adjacency)
    return result

  def clone(self):
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
    assert (self.num_graphs == 1)
    self.graph_nodes[self.offset] += 1
    self._adjacency.append([])
    self._node_tensor = torch.cat(
      (self._node_tensor[:self.nodes_including(self.offset)],
       node_tensor.unsqueeze(0).unsqueeze(0),
       self._node_tensor[self.nodes_including(self.offset):]), 0)
    return self._node_tensor.size(0) - 1

  def add_edge(self, source, target):
    self._adjacency[source].append(target)
    self._adjacency[target].append(source)
    return len(self._adjacency[source]) - 1

  def add_edges(self, edges):
    for edge in edges:
      self.add_edge(*edge)

  def delete_nodes(self, nodes):
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
    self.delete_nodes([node])

  def delete_edge(self, source, target):
    self._adjacency[source] = [x in self._adjacency[source] if x != target]
    self._adjacency[target] = [x in self._adjacency[target] if x != source]

  def delete_edges(self, edges):
    for edge in edges:
      self.delete_edge(*edge)

  def __getitem__(self, idx):
    assert (self.num_graphs > 1)
    # TODO
    return None

  def append(self, graph_tensor):
    assert(self.offset == 0)
    self.num_graphs += graph_tensor.num_graphs
    self.adjacency += list(map(
      lambda x: x + len(self._adjacency), graph_tensor.adjacency))
    self.graph_nodes += graph_tensor.graph_nodes
    self._node_tensor = torch.cat((self.node_tensor, graph_tensor.node_tensor), 0)

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

def batch_graphs(graphs):
  result = deepcopy(graph[0])
  for idx in range(1, len(graphs)):
    result.append(graphs[idx])
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

def standard_node_traversal(depth):
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
  def __init__(self, reducer, traversal=standard_node_traversal(1), order=None):
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

def NodeNeighbourhoodAttention(attention, traversal=standard_node_traversal(1)):
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
    new_node_tensor = local_attention + torch.spmm(adjacency, neighbour_attention)
    return new_node_tensor
  return reducer

def NodeNeighbourhoodSparseAttention(size, traversal=standard_node_traversal(1)):
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

def NodeNeighbourhoodDotAttention(size, traversal=standard_node_traversal(1)):
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

def NodeNeighbourhoodMean(traversal=standard_node_traversal(1)):
  """Aggregates a node neighbourhood using the mean of neighbour features."""
  return NodeGraphNeighbourhood(
    _node_neighbourhood_reducer(torch.Tensor.mean),
    traversal=traversal
  )

def NodeNeighbourhoodSum(traversal=standard_node_traversal(1)):
  """Aggregates a node neighbourhood using the sum of neighbour features."""
  return NodeGraphNeighbourhood(
    _node_neighbourhood_reducer(torch.Tensor.sum),
    traversal=traversal
  )

def NodeNeighbourhoodMax(traversal=standard_node_traversal(1)):
  """Aggregates a node neighbourhood using the maximum of neighbour features."""
  return NodeGraphNeighbourhood(
    _node_neighbourhood_reducer(torch.Tensor.max),
    traversal=traversal
  )

def NodeNeighbourhoodMin(traversal=standard_node_traversal(1)):
  """Aggregates a node neighbourhood using the minimum of neighbour features."""
  return NodeGraphNeighbourhood(
    _node_neighbourhood_reducer(torch.Tensor.min),
    traversal=traversal
  )

class ColorPool(nn.Module):
  def __init__(self, pooling, order=None,
               coloring=standard_node_coloring(2),
               traversal=standard_node_traversal(1)):
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
