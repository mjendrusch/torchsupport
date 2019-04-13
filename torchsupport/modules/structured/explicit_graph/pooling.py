import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.structured.graphnn import StandardNodeTraversal

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
      for node, edges in enumerate(graph.adjacency)
      for edge in edges
    )
    lookup = [
      []
      for node, edges in enumerate(graph.adjacency)
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
      sum(map(len, graph.adjacency[graph.graph_slice(idx)]))
      for idx in range(graph.num_graphs)
    ]
    new_nodes = torch.cat(new_nodes, dim=0)
    result.num_graphs = graph.num_graphs
    result.graph_nodes = new_graph_nodes
    result.adjacency = new_adjacency
    result.node_tensor = new_nodes
    return result

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
      _, indices = torch.topk(
        attention.node_tensor[graph.graph_range(idx)],
        graph_nodes // 2
      )
      all_topk.append(indices + graph_sum)
      graph_sum += graph_nodes
    all_topk = torch.cat(all_topk, dim=0).reshape(-1)
    attention.node_tensor = self.attention_activation(attention.node_tensor)
    attended = self.activation(graph * abs(attention) + graph)
    self.chosen = all_topk
    return self.color_pool(attended)

def MinimumDegreeNodeColoring():
  """Partitions a graph using a heuristic choosing all nodes with non-minimum
  connectivity in their neighbourhood.
  """
  def color(graph):
    chosen = []
    for idx in range(len(graph.adjacency)):
      lengths = [len(graph.adjacency[edge]) for edge in graph.adjacency[idx]]
      self_length = len(graph.adjacency[idx])
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
    values = func.normalize(torch.randn(graph.node_tensor.size(0), 1), dim=0)
    for _ in range(n_iter):
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
      for idx, node in enumerate(graph.adjacency[skip])
      for target in graph.adjacency[skip][idx+1:]
      if (node not in nodes_to_delete) and (target not in nodes_to_delete)
    ]

  def _ordered_neighbourhoods(self, graph, selected_nodes):
    neighbourhoods = []
    for node in selected_nodes:
      nodes = self.traversal(graph, node)
      if self.order is not None:
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
    out.node_tensor = torch.cat([
      self.pooling(graph.node_tensor[[node] + neighbourhoods[idx]])
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
    super(ColorUnpool, self).__init__()
    self.unpool = unpool

  def forward(self, graph, indices, guide_graph):
    out = guide_graph.new_like()
    out.node_tensor = torch.zeros((guide_graph.size(0), *graph.size[1:]))
    for node, edges in enumerate(guide_graph.adjacency):
      if node in indices:
        out.node_tensor[node] = graph.node_tensor[indices.index(node)]
        for target in edges:
          out.node_tensor[target] = self.unpool(
            out.node_tensor[target],
            graph.node_tensor[indices.index(node)]
          )
    return out
