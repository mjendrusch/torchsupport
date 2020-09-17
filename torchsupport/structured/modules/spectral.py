import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.structured.modules.basic import ConnectedModule
from torchsupport.structured import scatter

class AdjacencyAction(ConnectedModule):
  r"""Computes the action of the adjacency matrix defined by a given
  structure in the framework of message-passing neural networks.

  Args:
    normalized (bool): normalize contribution of the central node to its
      neighbours?
  """
  def __init__(self, normalized=False):
    super().__init__(has_scatter=True)

  def reduce_scatter(self, own_data, source_message, indices, node_count):
    _, counts = indices.unique(return_counts=True)
    message_size = torch.repeat_interleave(counts, counts).float()
    own_norm = message_size * (message_size + 1)
    return scatter.add(
      own_data / own_norm + source_message / (message_size + 1),
      indices, dim_size=node_count
    )

  def reduce(self, data, message):
    return (data + message.sum(dim=0)) / (message.size(0) + 1)

class LaplacianAction(ConnectedModule):
  r"""Computes the action of the graph Laplacian defined by a given
  structure in the framework of message-passing neural networks.

  Args:
    normalized (bool): normalize contribution of the central node to its
      neighbours?
  """
  def __init__(self, normalized=False):
    super(LaplacianAction, self).__init__(has_scatter=True)
    self.normalized = normalized

  def reduce_scatter(self, own_data, source_message, indices, node_count):
    _, counts = indices.unique(return_counts=True)
    message_size = torch.repeat_interleave(counts, counts).float()
    factor = 1
    if self.normalized:
      factor = 1 / message_size
    return scatter.add(
      (own_data - source_message) * factor,
      indices, dim_size=node_count
    )

  def reduce(self, data, message):
    factor = 1
    if self.normalized:
      factor = 1 / message.size(0)
    return factor * (message.size(0) * data - message.sum(dim=0))

class GCN(nn.Module):
  r"""Standard Graph Convolutional Neural Network (GCN).
  Transforms graph features by applying a non-linear transformation to
  all node features, followed by acting with the adjacency matrix, resulting
  in the following transformation:
  :math:`GCN(X, S) := A_S^d \sigma(wX + \mathbf{b})`
  where :math:`\sigma` is a nonlinear activation function, :math:`w` and
  :math:`b` the weights of an affine transformation, :math:`A_S` the
  adjacency matrix corresponding to the structure :math:`S` and
  :math:`d` the depth parameter of the GCN.

  Args:
    in_size (int): number of input feature maps.
    out_size (int): number of output feature maps.
    depth (int): exponent of the adjacency matrix action. Default: 1.
    activation (callable): nonlinear activation function.
  """
  def __init__(self, in_size, out_size, depth=1, activation=func.relu):
    super(GCN, self).__init__()
    self.linear = nn.Linear(in_size, out_size)
    self.connected = AdjacencyAction()
    self.activation = activation
    self.depth = depth

  def forward(self, data, structure):
    out = self.linear(data)
    for _ in range(self.depth):
      out = self.connected(out, out, structure)
    return self.activation(out)

class Chebyshev(nn.Module):
  r"""Chebyshev Graph Convolutional Neural Network.
  Transforms graph features by applying a non-linear transformation to
  all node features, followed by repeatedly acting with the graph Laplacian
  matrix.

  Args:
    in_size (int): number of input feature maps.
    out_size (int): number of output feature maps.
    depth (int): order of the Chebyshev polynomial approximation. Default: 1.
    activation (callable): nonlinear activation function.
  """
  def __init__(self, in_size, out_size, depth=1, activation=func.relu):
    super(Chebyshev, self).__init__()
    self.linear = nn.Linear(in_size, out_size)
    self.connected = LaplacianAction(normalized=True)
    self.activation = activation
    self.depth = depth

  def forward(self, data, structure):
    out_2 = self.linear(data)
    out_1 = self.connected(out_2, out_2, structure)
    out = out_1 + out_2
    for _ in range(self.depth):
      tmp = out_1
      out_1 = 2 * self.connected(out_1, out_1, structure) - out_2
      out_2 = tmp
      out += out_1
    return self.activation(out)

class ConvSkip(nn.Module):
  r"""Graph Convolutional Neural Network with skip connections.
  Transforms graph features by applying a non-linear transformation to
  all node features, followed by acting with the adjacency matrix or graph
  Laplacian and adding node features through a skip connection:
  :math:`Skip(X, Y, S) := \sigma(A_S f(X) + g(Y))`
  where :math:`\sigma` is a nonlinear activation function, :math:`f` and
  :math:`g` learnable affine transformations and :math:`A_S` the
  adjacency matrix corresponding to the structure :math:`S`.

  Args:
    in_size (int): number of input feature maps.
    out_size (int): number of output feature maps.
    merge_size (int): number of feature maps in the skip connection.
    activation (callable): nonlinear activation function.
    connected (:class:ConnectedModule): module computing the action
      of the adjacency matrix or graph Laplacian on transformed node features.
  """
  def __init__(self, in_size, out_size,
               merge_size, activation=func.relu,
               connected=LaplacianAction(normalized=True)):
    super(ConvSkip, self).__init__()
    self.transform = nn.Linear(merge_size, out_size)
    self.linear = nn.Linear(in_size, out_size)
    self.activation = activation
    self.connected = connected

  def forward(self, data, merge, structure):
    out = self.linear(data)
    out = self.connected(out, out, structure)
    return self.activation(out + self.transform(merge))

class WideConvSkip(nn.Module):
  def __init__(self, in_size, out_size, merge_size,
               width=3, activation=func.relu,
               connected=LaplacianAction(normalized=True)):
    super(WideConvSkip, self).__init__()
    self.transform = nn.Linear(merge_size, out_size * width)
    self.linear = nn.Conv1d(in_size * width, out_size * width, 1, groups=width)

  def forward(self, data, merge, structure):
    out = self.linear(data.unsqueeze(1)).squeeze()
    out = self.connected(out, out, structure)
    return self.activation(out + self.transform(merge))

class ARMA(nn.Module):
  def __init__(self, in_size, out_size, hidden_size,
               width=3, depth=3, share=False, activation=func.relu,
               connected=LaplacianAction(normalized=True)):
    super(ARMA, self).__init__()
    self.width = width
    self.preprocess = ConvSkip(
      in_size, hidden_size * width, in_size,
      connected=connected, activation=activation
    )

    if share:
      shared_block = WideConvSkip(
        hidden_size, hidden_size, in_size,
        width=width, connected=connected,
        activation=activation
      )
      self.blocks = nn.ModuleList([shared_block for _ in range(depth - 2)])
    else:
      self.blocks = nn.ModuleList([
        WideConvSkip(
          hidden_size, hidden_size, in_size,
          width=width, connected=connected,
          activation=activation
        )
        for _ in range(depth - 2)
      ])
    self.postprocess = WideConvSkip(
      hidden_size, out_size, in_size,
      width=width, connected=connected,
      activation=activation
    )

  def forward(self, data, structure):
    out = self.preprocess(data, data, structure)
    for block in self.blocks:
      out = block(out, data, structure)
    out = self.postprocess(out, data, structure)
    out = out.reshape(data.size(0), -1, self.width)
    return func.adaptive_avg_pool1d(out, 1)

class APP(nn.Module):
  def __init__(self, in_size, out_size,
               depth=10, teleport=0.5, activation=func.relu,
               connected=LaplacianAction(normalized=True)):
    super(APP, self).__init__()
    self.linear = nn.Linear(in_size, out_size)
    self.teleport = teleport
    self.depth = depth
    self.activation = activation
    self.connected = connected

  def forward(self, data, structure):
    embedding = self.linear(data)
    out = embedding
    for _ in range(self.depth):
      out = (1 - self.teleport) * self.connected(out, out, structure)
      out += self.teleport * embedding
    return self.activation(out)

class MultiScaleAPP(nn.Module):
  def __init__(self, in_size, out_size,
               depth=10, teleports=[0.1, 0.2, 0.3],
               activation=func.relu,
               connected=LaplacianAction(normalized=True)):
    super(MultiScaleAPP, self).__init__()
    self.source_attention = nn.Linear(in_size, out_size)
    self.target_attention = nn.Linear(out_size, out_size)
    self.scales = nn.ModuleList([
      APP(
        in_size, out_size,
        depth=depth, teleport=teleport,
        activation=activation,
        connected=connected
      )
      for teleport in teleports
    ])

  def forward(self, data, structure):
    scales = [
      scale(data, structure)
      for scale in self.scales
    ]
    source_attention = self.source_attention(data)
    scale_attention = torch.softmax(torch.cat([
      self.target_attention(scale).dot(source_attention)
      for scale in self.scales
    ], dim=1), dim=1)
    scales = torch.cat([
      scale.unsqueeze(1)
      for scale in scales
    ], dim=1)
    return (scale_attention * scales).sum(dim=1)
