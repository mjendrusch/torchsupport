import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.structured.nodegraph import cat

class GCN(nn.Module):
  def __init__(self, in_channels, out_channels,
               augment=1, activation=nn.ReLU(), matrix_free=False):
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
    new_nodes = self.linear(graph.node_tensor)
    normalization = torch.Tensor([len(edges) + 1 for edges in self.adjacency], dtype="float")
    for _ in range(self.augment):
      new_nodes = (self.adjacency_action(new_nodes, matrix_free=self.matrix_free) + new_nodes)
      new_nodes /= normalization
    out.node_tensor = self.activation(new_nodes)
    return out

class MultiscaleGCN(nn.Module):
  def __init__(self, in_channels, out_channels, scales,
               activation=nn.ReLU, matrix_free=False):
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
    for module in self.modules:
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
    m2_nodes = self.linear(graph.node_tensor)
    m1_nodes = graph.laplacian_action(m2_nodes)
    out_nodes = m1_nodes + m2_nodes
    for _ in range(2, self.depth):
      m2_nodes = m1_nodes
      m1_nodes += 2 * graph.laplacian_action(
        m1_nodes, matrix_free=self.matrix_free, normalized=True
      ) - m2_nodes
      out_nodes += m1_nodes
    out = graph.new_like()
    out.node_tensor = self.activation(out_nodes)
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
    nodes = graph.node_tensor

    out = self.preprocess(nodes)
    out = out.reshape(out.size(0), out.size(1) * out.size(2), 1)
    out += self.merge(nodes).reshape(out.size(0), out.size(1) * out.size(2), 1)
    out = self.activation(out)
    for _ in range(self.depth - 1):
      out -= graph.laplacian_action(out)
      out = self.propagate(out)
      out += self.merge(nodes).reshape(out.size(0), out.size(1) * out.size(2), 1)
      out = self.activation(out)

    out = out.reshape(nodes.size(0), nodes.size(1), self.width)
    out = func.adaptive_avg_pool1d(out, 1).reshape(
      nodes.size(0), -1
    ).unsqueeze(2)
    result = graph.new_like()
    result.node_tensor = out
    return result
