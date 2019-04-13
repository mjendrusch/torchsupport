import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.structured.nodegraph import PartitionedNodeGraphTensor, cat

class NodeModule(nn.Module):
  def node_update(self, node_tensor):
    raise NotImplementedError("Abstract")

  def forward(self, graph):
    out = graph.new_like()
    if isinstance(out, PartitionedNodeGraphTensor) and out.partition_view is not None:
      out.node_tensor = graph.node_tensor.clone()
      partition = out.partition[out.partition_view]
      out.node_tensor[partition, :] = self.node_update(out.node_tensor[partition, :])
    else:
      out.node_tensor = self.node_update(out.node_tensor)
    return out

class Linear(NodeModule):
  def __init__(self, insize, outsize):
    """Applies a linear module at each node in a graph.

    Args:
      insize (int): number of input features.
      outsize (int): number of output features.

    Returns:
      `NodeModule` wrapping a `nn.Linear` module.
    """
    super(Linear, self).__init__()
    self.linear = nn.Linear(insize, outsize)

  def node_update(self, node_tensor):
    node_tensor = node_tensor.view(node_tensor.size()[0], -1)
    node_tensor = self.linear(node_tensor)
    node_tensor = node_tensor.unsqueeze(1)
    return node_tensor

class NodeTraversal(object):
  def __init__(self, packed=True):
    self.packed = packed

  def traverse(self, graph, node):
    raise NotImplementedError("Abstract")

  def extract(self, graph, neighbourhood):
    if self.packed:
      return graph.node_tensor[neighbourhood]
    return neighbourhood

  def __call__(self, graph):
    partition = range(graph.node_tensor.size(0))
    if isinstance(graph, PartitionedNodeGraphTensor) and graph.partition_view is not None:
      partition = graph.partition[graph.partition_view]
    for node in partition:
      nodes = self.traverse(graph, node)
      yield node, self.extract(graph, nodes)

class StandardNodeTraversal(NodeTraversal):
  def __init__(self, depth, with_self=True):
    """Computes a standard node traversal for a given node.

    Args:
      depth (int): number of hops in the n-hop traversal.
      with_self (bool): include the origin of the traversal? Defaults to `True`.
    """
    super(StandardNodeTraversal, self).__init__()
    self.depth = depth
    self.with_self = with_self

  @staticmethod
  def _traverse_aux(graph, entity, with_self, depth):
    if depth == 0:
      return [entity] if with_self else []
    nodes = [entity] if with_self else []
    edges = graph.adjacency[entity]
    nodes += edges
    for new_node in edges:
      if new_node != entity:
        new_nodes = StandardNodeTraversal._traverse_aux(
          graph, new_node, with_self, depth - 1
        )
        nodes += new_nodes
    nodes = list(set(nodes))
    return nodes

  def traverse(self, graph, node):
    return StandardNodeTraversal._traverse_aux(graph, node, self.with_self, self.depth)

class CloseDisconnectedNodeTraversal(NodeTraversal):
  def __init__(self, reject_depth, radius, position_slice=slice(0, 3)):
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
    super(CloseDisconnectedNodeTraversal, self).__init__()
    self.reject_depth = reject_depth
    self.radius = radius
    self.position_slice = position_slice
    self.reject_node_traversal = StandardNodeTraversal(reject_depth)

  def traverse(self, graph, entity):
    reject_nodes = self.reject_node_traversal.traverse(graph, entity)
    accept_nodes = []
    graph_slice = graph.graph_slice(graph.node_graph(entity))
    radius = torch.norm(
      graph.node_tensor[graph_slice][self.position_slice] - graph.node_tensor[entity]
    )
    for node, distance in enumerate(radius, graph_slice.start):
      if distance < radius and node not in reject_nodes:
        accept_nodes.append(node)
    return entity, accept_nodes

class DynamicAttentionNodeTraversal(NodeTraversal):
  def __init__(self, top_p=0.8):
    """Computes a dynamic, attention-guided node neighbourhood.

    Args:
      attention_left (NodeGraphTensor): node-part of dot-product attention.
      attention_right (NodeGraphTensor): neighbourhood-part of dot-product attention.
      top_p (float): percentage of signal to cover.

    Returns:
      Nodes assigned to a given origin node by global dot-attention.
    """
    super(DynamicAttentionNodeTraversal, self).__init__()
    self.top_p = top_p
    self.attention_map = None

  def extract(self, input_graphs, neighbourhood):
    result = self.attention_map
    self.attention_map = None
    if self.packed:
      return (input_graphs[0][neighbourhood], result)
    return neighbourhood, result

  def traverse(self, inputs, entity):
    graph, attention_left, attention_right = inputs
    entity_slice = graph.graph_slice(graph.node_graph(entity))
    self.attention_map = attention_left.node_tensor.dot(
      attention_right.node_tensor[graph.graph_slice(graph.node_graph(entity))]
    )
    self.attention_map = func.softmax(self.attention_map)
    sorted_map, indices = torch.sort(self.attention_map, descending=True)
    total_percentage = 0
    index = 0
    while total_percentage < self.top_p:
      total_percentage += sorted_map[index]
      index += 1
    return [index + entity_slice.start for index in list(indices[:index + 1])]

class NeighbourModule(nn.Module):
  def __init__(self, traversal):
    """Applies a reduction function to the neighbourhood of each node.

    Args:
      traversal (callable): node traversal for generating node neighbourhoods.
    """
    super(NeighbourModule, self).__init__()
    self.traversal = traversal

  def reduce(self, graph, neighbourhood):
    raise NotImplementedError("Abstract")

  def forward(self, graph):
    out = self.reduce(graph, self.traversal(graph))
    return out

class NeighbourAssignment(NeighbourModule):
  def __init__(self, in_channels, out_channels, size,
               traversal=StandardNodeTraversal(1)):
    """Aggregates a node neighbourhood using soft weight assignment. (FeaStNet)

    Args:
      in_channels (int): number of input features.
      out_channels (int): number of output features.
      size (int): number of distinct weight matrices (equivalent to kernel size).
      traversal (callable): node traversal for generating node neighbourhoods.
    """
    super(NeighbourAssignment, self).__init__(traversal)
    self.linears = nn.ModuleList([
      nn.Linear(in_channels, out_channels)
      for _ in range(size)
    ])
    self.source = nn.Linear(in_channels, size)
    self.target = nn.Linear(in_channels, size)

  def reduce(self, graph, neighbourhood):
    source_tensor = self.source(graph.node_tensor)
    target_tensor = self.target(graph.node_tensor)
    weight_tensors = []
    for module in self.linears:
      weight_tensors.append(module(graph.node_tensor).unsqueeze(0))
    weighted = torch.cat(weight_tensors, dim=0)
    newnode_tensor = torch.zeros_like(graph.node_tensor)
    for idx, node in neighbourhood:
      assignment_tensor = func.softmax(source_tensor[idx] + target_tensor[node]).unsqueeze(0)
      newnode_tensor[idx] = (assignment_tensor * weighted[node]).mean(dim=0).squeeze(0)
    out = graph.new_like()
    out.node_tensor = newnode_tensor
    return out

class NeighbourAttention(NeighbourModule):
  def __init__(self, attention, traversal=StandardNodeTraversal(1)):
    """Aggregates a node neighbourhood using an attention mechanism.

    Args:
      attention (callable): attention mechanism to be used.
      traversal (callable): node traversal for generating node neighbourhoods.
    """
    super(NeighbourAttention, self).__init__(traversal)
    self.attention = attention

  def reduce(self, graph, neighbourhood):
    newnode_tensor = torch.zeros_like(graph.node_tensor)
    for idx, node in neighbourhood:
      local_tensor = graph.node_tensor[node]
      attention = self.attention(torch.cat((local_tensor, graph.node_tensor[idx])))
      reduced = torch.Tensor.sum(attention * local_tensor, dim=1)
      newnode_tensor[idx] = reduced if isinstance(reduced, torch.Tensor) else reduced[0]
    out = graph.new_like()
    out.node_tensor = newnode_tensor
    return out

def neighbourhood_to_adjacency(neighbourhood):
  size = torch.Size([len(neighbourhood), len(neighbourhood)])
  indices = []
  for idx, nodes in neighbourhood:
    for node in nodes:
      indices.append([idx, node])
      indices.append([node, idx])
  indices = torch.Tensor(list(set(indices)))
  values = torch.ones(indices.size(0))
  return torch.sparse_coo_tensor(indices, values, size)

class NeighbourSparseAttention(NeighbourModule):
  def __init__(self, size, traversal=StandardNodeTraversal(1)):
    """Aggregates a node neighbourhood using a sparse attention mechanism.

    Args:
      size (int): size of the attention embedding.
      traversal (callable): node traversal for generating node neighbourhoods.
    """
    super(NeighbourSparseAttention, self).__init__(traversal)
    self.embedding = nn.Linear(size, size)
    self.attention_local = nn.Linear(size, 1)
    self.attention_neighbour = nn.Linear(size, 1)

  def reduce(self, graph, neighbourhood):
    embedding = self.embedding(graph.node_tensor)
    local_attention = self.attention_local(embedding)
    neighbour_attention = self.attention_neighbour(embedding)
    adjacency = neighbourhood_to_adjacency(neighbourhood)
    newnode_tensor = local_attention + torch.mm(adjacency, neighbour_attention)
    out = graph.new_like()
    out.node_tensor = newnode_tensor
    return out

class NeighbourDotAttention(NeighbourModule):
  def __init__(self, size, traversal=StandardNodeTraversal(1)):
    """Aggregates a node neighbourhood using a pairwise dot-product attention mechanism.
    Args:
      size (int): size of the attention embedding.
      traversal (callable): node traversal for generating node neighbourhoods.
    """
    super(NeighbourDotAttention, self).__init__(traversal)
    self.embedding = nn.Linear(size, size)
    self.attention_local = nn.Linear(size, 1)
    self.attention_neighbour = nn.Linear(size, 1)

  def reduce(self, graph, neighbourhood):
    embedding = self.embedding(graph.node_tensor)
    local_attention = self.attention_local(embedding)
    neighbour_attention = self.attention_neighbour(embedding)
    newnode_tensor = torch.zeros_like(graph.node_tensor)
    for idx, node in neighbourhood:
      reduced = torch.Tensor.sum(
        (local_attention[idx] + neighbour_attention[node]) * graph.node_tensor[node],
        dim=1
      )
      newnode_tensor[idx] = reduced if isinstance(reduced, torch.Tensor) else reduced[0]
    out = graph.new_like()
    out.node_tensor = newnode_tensor
    return out

class NeighbourReducer(NeighbourModule):
  def __init__(self, reduction, traversal=StandardNodeTraversal(1)):
    super(NeighbourReducer, self).__init__(traversal)
    self.reduction = reduction

  def reduce(self, graph, neighbourhood):
    newnode_tensor = torch.zeros_like(graph.node_tensor)
    for idx, node in neighbourhood:
      reduced = self.reduction(graph.node_tensor[node], dim=1)
      newnode_tensor[idx] = reduced if isinstance(reduced, torch.Tensor) else reduced[0]
    out = graph.new_like()
    out.node_tensor = newnode_tensor
    return out

class NeighbourMean(NeighbourReducer):
  def __init__(self, traversal=StandardNodeTraversal(1)):
    super(NeighbourMean, self).__init__(torch.mean, traversal=traversal)

class NeighbourSum(NeighbourReducer):
  def __init__(self, traversal=StandardNodeTraversal(1)):
    super(NeighbourSum, self).__init__(torch.sum, traversal=traversal)

class NeighbourMin(NeighbourReducer):
  def __init__(self, traversal=StandardNodeTraversal(1)):
    super(NeighbourMin, self).__init__(torch.min, traversal=traversal)

class NeighbourMax(NeighbourReducer):
  def __init__(self, traversal=StandardNodeTraversal(1)):
    super(NeighbourMax, self).__init__(torch.max, traversal=traversal)

class NeighbourMedian(NeighbourReducer):
  def __init__(self, traversal=StandardNodeTraversal(1)):
    super(NeighbourMedian, self).__init__(torch.median, traversal=traversal)

class GraphResBlock(nn.Module):
  def __init__(self, channels, aggregate=NeighbourMax,
               activation=nn.ReLU()):
    """Residual block for graph networks.

    Args:
      channels (int): number of input and output features.
      aggregate (nn.Module): neighbourhood aggregation function.
      activation (nn.Module): activation function. Defaults to ReLU.
    """
    super(GraphResBlock, self).__init__()
    self.activation = activation
    self.aggregate = aggregate
    self.linear = Linear(2 * channels, channels)

  def forward(self, graph):
    out = self.aggregate(graph)
    out = self.linear(out)
    out = self.activation(out + graph)
    return out

