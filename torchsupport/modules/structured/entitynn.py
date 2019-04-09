import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.structured.connected_entities import EntityTensor, AdjacencyStructure

class LocalModule(nn.Module):
  def local_update(self, *inputs):
    raise NotImplementedError("Abstract")

  def forward(self, *inputs):
    inp = map(lambda x: x.current_view, inputs)
    outputs = self.local_update(*inp)
    out = inputs[0].new_like()
    out.current_view = outputs
    return out

class Linear(LocalModule):
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

  def local_update(self, node_tensor):
    node_tensor = node_tensor.view(node_tensor.size(0), -1)
    node_tensor = self.linear(node_tensor)
    return node_tensor

class ConnectedModule(nn.Module):
  def __init__(self):
    """Applies a reduction function to the neighbourhood of each entity."""
    super(ConnectedModule, self).__init__()

  def reduce(self, own_data, source_messages):
    raise NotImplementedError("Abstract")

  def forward(self, entity_tensor, structure):
    out = entity_tensor.new_like()
    out = out.view(structure.target)
    results = []
    for idx, message in enumerate(structure.message(entity_tensor)):
      reduced = self.reduce(entity_tensor.current_view[idx], message)
      results.append(reduced.unsqueeze(0))
    out.current_view = torch.cat(results, dim=0)
    return out

class NeighbourLinear(ConnectedModule):
  def __init__(self, source_channels, target_channels):
    super(NeighbourLinear).__init__()
    self.linear = nn.Linear(source_channels, target_channels)

  def reduce(self, own_data, source_messages):
    return own_data + func.relu(self.linear(source_messages)).mean(dim=0, keepdim=True)

class NeighbourAssignment(ConnectedModule):
  def __init__(self, source_channels, target_channels, out_channels, size):
    """Aggregates a node neighbourhood using soft weight assignment. (FeaStNet)

    Args:
      in_channels (int): number of input features.
      out_channels (int): number of output features.
      size (int): number of distinct weight matrices (equivalent to kernel size).
    """
    super(NeighbourAssignment, self).__init__()
    self.linears = nn.ModuleList([
      nn.Linear(source_channels, out_channels)
      for _ in range(size)
    ])
    self.source = nn.Linear(source_channels, size)
    self.target = nn.Linear(target_channels, size)

  def reduce(self, own_data, source_message, idx=None):
    target = self.target(own_data)
    source = self.source(source_message)
    weight_tensors = []
    for module in self.linears:
      weight_tensors.append(module(source_message).unsqueeze(0))
    weighted = torch.cat(weight_tensors, dim=0)
    assignment = func.softmax(source + target).unsqueeze(0)
    return (assignment * weighted).mean(dim=0)

class NeighbourAttention(ConnectedModule):
  def __init__(self, attention):
    """Aggregates a node neighbourhood using an attention mechanism.

    Args:
      attention (callable): attention mechanism to be used.
      traversal (callable): node traversal for generating node neighbourhoods.
    """
    super(NeighbourAttention, self).__init__()
    self.attention = attention

  def reduce(self, own_data, source_message):
    attention = self.attention(own_data, source_message)
    result = torch.Tensor.sum(attention * source_message, dim=0)
    return result

class NeighbourDotAttention(ConnectedModule):
  def __init__(self, size):
    """Aggregates a node neighbourhood using a pairwise dot-product attention mechanism.
    Args:
      size (int): size of the attention embedding.
    """
    super(NeighbourDotAttention, self).__init__()
    self.embedding = nn.Linear(size, size)
    self.attention_local = nn.Linear(size, 1)
    self.attention_neighbour = nn.Linear(size, 1)

  def reduce(self, own_data, source_message):
    target = self.attention_local(self.embedding(own_data))
    source = self.attention_neighbour(self.embedding(source_message))
    result = ((target + source) * source_message).sum(dim=0)
    return result

class NeighbourReducer(ConnectedModule):
  def __init__(self, reduction):
    super(NeighbourReducer, self).__init__()
    self.reduction = reduction

  def reduce(self, own_data, source_message):
    return self.reduction(source_message, dim=0)

class NeighbourMean(NeighbourReducer):
  def __init__(self):
    super(NeighbourMean, self).__init__(torch.mean)

class NeighbourSum(NeighbourReducer):
  def __init__(self):
    super(NeighbourSum, self).__init__(torch.sum)

class NeighbourMin(NeighbourReducer):
  def __init__(self):
    super(NeighbourMin, self).__init__(torch.min)

class NeighbourMax(NeighbourReducer):
  def __init__(self):
    super(NeighbourMax, self).__init__(torch.max)

class NeighbourMedian(NeighbourReducer):
  def __init__(self):
    super(NeighbourMedian, self).__init__(torch.median)

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
