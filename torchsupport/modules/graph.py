import torch
import torch.nn as nn
import torch.nn.functional as func
from copy import copy

class GraphTensor(object):
  def __init__(self, graphdesc=None):
    self.maximum_edge_size = None

    self.is_subgraph = False
    self.offset = 0

    if graphdesc == None:
      self.num_graphs = 1
      self.graph_nodes = [0]
      self.graph_edges = [0]
      self.graph_globals = [0]
      self._nodes = []
      self._edges = []

      self._node_tensor = torch.tensor([])
      self._edge_tensor = torch.tensor([])
      self._global_tensor = torch.tensor([])
    else:
      self.num_graphs = graphdesc["num_graphs"]
      self.graph_nodes = graphdesc["graph_nodes"]
      self.graph_edges = graphdesc["graph_edges"]
      self.graph_globals = graphdesc["graph_globals"]

      self._nodes = graphdesc["nodes"]
      self._edges = graphdesc["edges"]

      self._node_tensor = graphdesc["node_tensor"]
      self._edge_tensor = graphdesc["edge_tensor"]
      self._global_tensor = graphdesc["global_tensor"]

  @property
  def is_graph(self):
    return self.maximum_edge_size == 2

  def subgraph(self, idx):
    assert(idx < self.num_graphs)
    view = copy(self)
    view.is_subgraph = True
    view.offset = idx
    return view

  @property
  def parent_graph(self):
    view = copy(self)
    view.is_subgraph = False
    view.offset = 0
    return view

  @property
  def nodes(self):
    start = 0 if self.offset == 0 else self.nodes_including(self.offset - 1)
    stop = self.nodes_including(self.offset)
    return self._nodes[start:stop]

  @property
  def edges(self):
    start = 0 if self.offset == 0 else self.edges_including(self.offset - 1)
    stop = self.edges_including(self.offset)
    return self._edges[start:stop]

  @property
  def node_tensor(self):
    start = 0 if self.offset == 0 else self.nodes_including(self.offset - 1)
    stop = self.nodes_including(self.offset)
    return self._node_tensor[start:stop]

  @property
  def edge_tensor(self):
    start = 0 if self.offset == 0 else self.edges_including(self.offset - 1)
    stop = self.edges_including(self.offset)
    return self._edge_tensor[start:stop]

  def incoming(self, node):
    assert(self.is_graph())
    start = 0 if self.offset == 0 else self.edges_including(self.offset - 1)
    return (edge - start for edge in node if node == self._edges[edge][1])

  def outgoing(self, node):
    assert(self.is_graph())
    start = 0 if self.offset == 0 else self.edges_including(self.offset - 1)
    return (edge - start for edge in node if node == self._edges[edge][1])

  def nodes_including(self, graph_index):
    return sum(self.graph_nodes[:graph_index])

  def edges_including(self, graph_index):
    return sum(self.graph_edges[:graph_index])

  def add_node(self, node_tensor):
    self._nodes.insert(self.nodes_including(self.offset), [])
    self.graph_nodes[self.offset] += 1
    self._node_tensor = torch.cat(
      (self._node_tensor[:self.nodes_including(self.offset)],
       node_tensor.unsqueeze(0),
       self._node_tensor[self.nodes_including(self.offset):]), 0)
    return len(self._nodes) - 1

  def add_edge(self, edge_tensor, *nodes):
    self._edges.insert(self.edges_including(self.offset), nodes)
    for node in nodes:
      self._nodes[node].append(len(self._edges) - 1)
    self.graph_edges[self.offset] += 1
    self._edge_tensor = torch.cat(
      (self._edge_tensor[:self.edges_including(self.offset)],
       edge_tensor.unsqueeze(0),
       self._edge_tensor[self.edges_including(self.offset):]), 0)
    return len(self._edges) - 1

  def append(self, graph_tensor):
    assert(self.offset == 0)
    self.num_graphs += graph_tensor.num_graphs
    self.graph_edges += graph_tensor.graph_edges
    self.graph_nodes += graph_tensor.graph_nodes
    self.graph_globals += graph_tensor.graph_globals
    self.maximum_edge_size = max(self.maximum_edge_size, graph_tensor.maximum_edge_size)

    self._node_tensor = torch.cat((self.node_tensor, graph_tensor.node_tensor), 0)
    self._edge_tensor = torch.cat((self.edge_tensor, graph_tensor.edge_tensor), 0)
    self._global_tensor = torch.cat((self.global_tensor, graph_tensor.global_tensor), 0)

class EachNode(nn.Module):
  def __init__(self, node_update):
    super(self, EachNode).__init__(self)
    self.node_update = node_update

  def forward(self, graph, neighbour_embedding=None, global_embedding=None):
    additional_data = []
    if neighbour_embedding != None:
      additional_data.append(neighbour_embedding)
    if global_embedding != None:
      additional_data.append(global_embedding)
    graph.node_tensor = self.node_update(graph.node_tensor, *additional_data)
    return graph

class EachEdge(nn.Module):
  def __init__(self, edge_update):
    super(self, EachEdge).__init__(self)
    self.edge_update = edge_update

  def forward(self, graph, neighbour_embedding=None, global_embedding=None):
    additional_data = []
    if neighbour_embedding != None:
      additional_data.append(neighbour_embedding)
    if global_embedding != None:
      additional_data.append(global_embedding)
    graph.edge_tensor = self.edge_update(graph.edge_tensor, *additional_data)
    return graph

class NodeToGlobal(nn.Module):
  def __init__(self, reducer, order=None):
    super(self, NodeToGlobal).__init__(self)
    self.reducer = reducer
    self.order = order

  def forward(self, graph):
    nodes = graph.node_tensor
    if self.order != None:
      nodes = self.order(nodes)
    reduced_nodes = self.reducer(graph, nodes)
    graph.global_tensor = torch.cat((graph.global_tensor, reduced_nodes), 1)
    return graph

class EdgeToGlobal(nn.Module):
  def __init__(self, reducer, order=None):
    super(self, EdgeToGlobal).__init__(self)
    self.reducer = reducer
    self.order = order

  def forward(self, graph):
    edges = graph.edge_tensor
    if self.order != None:
      edges = self.order(edges)
    reduced_edges = self.reducer(graph, edges)
    graph.global_tensor = torch.cat((graph.global_tensor, reduced_edges), 1)
    return graph

def _standard_neighbourhood_traversal_aux(
  graph, entity, depth, collect_nodes, collect_edges, is_edge=False
):
  if depth == 0:
      return [entity], []
  else:
    if is_edge:
      edges = []
      nodes = graph.edges[entity]
      for node in nodes:
        edges += graph.nodes[node]
      for new_edge in edges:
        if new_edge != entity:
          new_edges, new_nodes = _standard_neighbourhood_traversal_aux(
            graph, new_edge, depth - 1, collect_nodes, collect_edges, is_edge
          )
          edges += new_edges
          nodes += new_nodes
      edges = list(set(edges)) if collect_edges else None
      nodes = list(set(nodes)) if collect_nodes else None
      return edges, nodes
    else:
      nodes = []
      edges = graph.nodes[entity]
      for edge in edges:
        nodes += graph.edges[edge]
      for new_node in nodes:
        if new_node != entity:
          new_nodes, new_edges = _standard_neighbourhood_traversal_aux(
            graph, new_edge, depth - 1, collect_nodes, collect_edges, is_edge
          )
          nodes += new_nodes
          edges += new_edges
      nodes = list(set(nodes)) if collect_nodes else None
      edges = list(set(edges)) if collect_edges else None
      return nodes, edges

def standard_node_neighbourhood_traversal(depth, collect_nodes=False, collect_edges=True):
  return lambda graph, node: _standard_neighbourhood_traversal_aux(
    graph, node, depth, collect_nodes, collect_edges, is_edge=False
  )

def standard_edge_neighbourhood_traversal(depth, collect_nodes=True, collect_edges=False):
  return lambda graph, edge: _standard_neighbourhood_traversal_aux(
    graph, edge, depth, collect_nodes, collect_edges, is_edge=True
  )

class GeneralNodeNeighbourhood(nn.Module):
  def __init__(self, reducer, traversal=standard_node_neighbourhood_traversal(1), order=None):
    super(self, GeneralNodeNeighbourhood).__init__(self)
    self.reducer = reducer
    self.traversal = traversal
    self.order = order

  def forward(self, graph, include_self=True):
    full_nodes = []
    full_edges = []
    for node, _ in enumerate(graph.nodes):
      nodes, edges = self.traversal(graph, node)
      if self.collect_nodes:
        sorted_nodes = self.order(graph, nodes)
        full_nodes.append(sorted_nodes)
      if self.collect_edges:
        sorted_edges = self.order(graph, edges)
        full_edges.append(sorted_edges)
    
    reduced_nodes = self.reducer(graph, full_nodes)
    reduced_edges = self.reducer(graph, full_edges)

    graph._node_tensor = torch.cat((graph._node_tensor, reduced_nodes, reduced_edges), 1)

    return graph

class GeneralEdgeNeighbourhood(object):
  def __init__(self, reducer, traversal=traversal=standard_edge_neighbourhood_traversal(1), order=None):
    super(self, GeneralEdgeNeighbourhood).__init__(self)
    self.reducer = reducer
    self.traversal = traversal
    self.order = order

  def forward(self, graph, include_self=True):
    full_edges = []
    full_nodes = []
    for edge, _ in enumerate(graph.edges):
      edges, nodes = self.traversal(graph, edge)
      if self.collect_edges:
        sorted_edges = self.order(graph, edges)
        full_edges.append(sorted_edges)
      if self.collect_nodes:
        sorted_nodes = self.order(graph, nodes)
        full_nodes.append(sorted_nodes)
    
    reduced_edges = self.reducer(graph, full_edges)
    reduced_nodes = self.reducer(graph, full_nodes)

    graph._edge_tensor = torch.cat((graph._edge_tensor, reduced_edges, reduced_nodes), 1)

    return graph
