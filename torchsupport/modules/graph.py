import torch
import torch.nn as nn
import torch.nn.functional as func

class Edge(object):
  def __init__(self, nodes, label, id=0):
    self.id = id
    self.nodes = nodes
    self.label = label

  def __len__(self):
    return len(nodes)

class Node(object):
  def __init__(self, edges, label, id=0):
    self.id = id
    self.edges = edges
    self.label = label

  def __len__(self):

class Graph(object):
  """
    Graph(nodes=[], edges=[])
  A (hyper-)graph made up of labelled nodes and labelled (hyper-)edges.
  """
  def __init__(self, edges=[], nodes=[]):
    self.edges = edges
    self.nodes = nodes
    self.current_node_id = 0
    self.current_edge_id = 0

  def add_node(self, label):
    self.nodes.append(Node([], label, self.current_node_id))
    self.current_node_id += 1
    return self.nodes[-1]

  def add_edge(self, label, *nodes):
    self.edges.append(Edge([*nodes], label, self.current_edge_id))
    self.current_edge_id += 1
    return self.edges[-1]

  def single_edge_nodes(self, edge):
    result = []
    for node in edge.nodes:
      result.append(node.id)
    return result

  def edge_nodes(self):
    result = []
    for edge in self.edges:
      result.append(self.single_edge_nodes(edge))
    return result

  def single_node_edges(self, node):
    result = []
    for edge in node.edges:
      result.append(edge.id)
    return result

  def node_edges(self):
    result = []
    for node in self.nodes:
      result.append(self.single_node_edges(node))
    return result

  def torch_graph(self, scalar_size=(1,)):
    """
      torch_graph(scalar_size=(1,))
    Creates a torch tensor graph from a Graph.
    """
    en = self.edge_nodes
    ne = self.node_edges
    ns = self.nodes[-1].label.size()
    es = self.edges[-1].label.size()
    ss = scalar_size
    return TorchGraph(en, ne, ns, es, ss)

class TorchGraph(object):
  """
    TorchGraph(edge_nodes, node_edges,
               node_size, edge_size, scalar_size)
  A graph for graph neural network consumption.
  """
  def __init__(self, edge_nodes, node_edges,
               node_size, edge_size, scalar_size):
    self.edge_nodes = edge_nodes
    self.node_edges = node_edges
    self.edge_tensor = torch.zeros(edge_size)
    self.node_tensor = torch.tensor(node_size)
    self.scalar = torch.tensor(scalar_size)

class GraphBlock(nn.Module):
  """
    GraphBlock(edge, node, scalar,
               edge_to_node, edge_to_scalar,
               node_to_edge, node_to_scalar)
  A (hyper-) graph neural network block.
  """
  def __init__(self, edge, node, scalar,
               edge_to_node, edge_to_scalar,
               node_to_edge, node_to_scalar):
    super(self, GraphBlock).__init__(self)
    self.edge = edge
    self.node = node
    self.scalar = scalar
    self.edge_to_node = edge_to_node
    self.edge_to_scalar = edge_to_scalar
    self.node_to_edge = node_to_edge
    self.node_to_scalar = node_to_scalar

  def forward(self, graph):
    graph = self.node_to_edge(graph)
    graph = self.edge(graph)
    graph = self.edge_to_node(graph)
    graph = self.edge_to_scalar(graph)
    graph = self.node(graph)
    graph = self.node_to_scalar(graph)
    return graph

# TODO
class GraphPooling(nn.Module):
  pass

# TODO
class GraphMonoidalFunctor(nn.Module):
  """
  Graph pooling obeying lax monoidal functor laws.
  """
  pass
