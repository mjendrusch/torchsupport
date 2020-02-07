import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.structured.structures import (
  ConstantStructure, ScatterStructure, MessageMode
)
from .. import scatter

def flatten_message(message):
  return message.view(-1, *message.shape[2:])

def unflatten_message(output, message):
  return output.view(*message.shape[:2], *output.shape[1:])

class ConnectedModule(nn.Module):
  def __init__(self, has_scatter=False):
    """Applies a reduction function to the neighbourhood of each entity."""
    super(ConnectedModule, self).__init__()
    self.has_scatter = has_scatter

  def reduce(self, own_data, source_messages):
    raise NotImplementedError("Abstract")

  def reduce_scatter(self, own_data, source_message, indices, node_count):
    raise NotImplementedError("Abstract")

  def forward(self, source, target, structure):
    # constant-width neighbourhoods:
    if structure.mode_is(MessageMode.constant):
      return self.reduce(target, structure.message(source, target))
    if structure.mode_is(MessageMode.scatter):
      if not self.has_scatter:
        raise NotImplementedError(
          "Scattering-based implementation not supported for {self.__class__.__name__}."
        )
      source, target, indices, node_count = structure.message(source, target)
      return self.reduce_scatter(target, source, indices, node_count)

    results = []
    for idx, message in enumerate(structure.message(source, target)):
      reduced = self.reduce(target[idx].unsqueeze(dim=0), message)
      results.append(reduced)
    return torch.cat(results, dim=0)

class NeighbourLinear(ConnectedModule):
  def __init__(self, source_channels, target_channels):
    super(NeighbourLinear, self).__init__(has_scatter=True)
    self.linear = nn.Linear(source_channels, target_channels)

  def reduce_scatter(self, own_data, source_message, indices, node_count):
    return scatter.mean(
      own_data + func.relu(self.linear(source_message)),
      indices, dim_size=node_count
    )

  def reduce(self, own_data, source_message):
    inputs = flatten_message(source_message)
    out = unflatten_message(self.linear(inputs), source_message)
    return own_data + func.relu(out).mean(dim=1)

class NeighbourAssignment(ConnectedModule):
  def __init__(self, source_channels, target_channels, out_channels, size):
    """Aggregates a node neighbourhood using soft weight assignment. (FeaStNet)

    Args:
      in_channels (int): number of input features.
      out_channels (int): number of output features.
      size (int): number of distinct weight matrices (equivalent to kernel size).
    """
    super(NeighbourAssignment, self).__init__(has_scatter=True)
    self.linears = nn.ModuleList([
      nn.Linear(source_channels, out_channels)
      for _ in range(size)
    ])
    self.source = nn.Linear(source_channels, size)
    self.target = nn.Linear(target_channels, size)

  def reduce_scatter(self, own_data, source_message, indices, node_count):
    target = self.target(own_data)
    source = self.source(source_message)
    weight_tensors = []
    for module in self.linears:
      weight_tensors.append(module(source_message).unsqueeze(0))
    weighted = torch.cat(weight_tensors, dim=0)
    assignment = func.softmax(source + target, dim=-1).unsqueeze(0)
    result = assignment.transpose(0, 3) * weighted
    return scatter.mean(result, dim_size=node_count)

  def reduce(self, own_data, source_message, idx=None):
    inputs = flatten_message(source_message)
    target = self.target(own_data).unsqueeze(0)
    source = self.source(inputs)
    weight_tensors = []
    for module in self.linears:
      result = unflatten_message(module(inputs), source_message)
      weight_tensors.append(result.unsqueeze(0))
    weighted = torch.cat(weight_tensors, dim=0)
    source = unflatten_message(source, source_message)
    assignment = func.softmax(source + target, dim=-1).unsqueeze(0)
    return (assignment.transpose(0, 3) * weighted).mean(dim=0)

class NeighbourAttention(ConnectedModule):
  def __init__(self, in_size, out_size, query_size=None, attention_size=None):
    """Aggregates a node neighbourhood using a pairwise dot-product attention mechanism.
    Args:
      size (int): size of the attention embedding.
    """
    super(NeighbourAttention, self).__init__(has_scatter=True)
    query_size = query_size if query_size is not None else in_size
    attention_size = attention_size if attention_size is not None else in_size
    self.query = nn.Linear(query_size, attention_size)
    self.key = nn.Linear(in_size, attention_size)
    self.value = nn.Linear(in_size, out_size)

  def attend(self, query, data):
    raise NotImplementedError("Abstract.")

  def reduce_scatter(self, own_data, source_message, indices, node_count):
    target = self.query(own_data)
    source = self.key(source_message)
    value = self.value(source_message)
    attention = scatter.softmax(
      self.attend(target, source), indices, dim_size=node_count
    )
    result = scatter.add(
      (attention.unsqueeze(-1) * value), indices, dim_size=node_count
    )
    return result

  def reduce(self, own_data, source_message):
    target = self.query(own_data)
    inputs = flatten_message(source_message)
    source = self.key(inputs)
    value = self.value(inputs)
    source = unflatten_message(source, source_message)
    value = unflatten_message(value, source_message)

    attention = func.softmax(self.attend(target.unsqueeze(1), source), dim=1)
    result = (attention.unsqueeze(-1) * value).sum(dim=1)
    return result

class NeighbourDotAttention(NeighbourAttention):
  def attend(self, query, data):
    return (query * data).sum(dim=-1)

class NeighbourAddAttention(NeighbourAttention):
  def attend(self, query, data):
    return (query + data).sum(dim=-1)

class NeighbourMultiHeadAttention(ConnectedModule):
  def __init__(self, in_size, out_size, attention_size, query_size=None, heads=64,
               normalization=lambda x: x):
    """Aggregates a node neighbourhood using a pairwise dot-product attention mechanism.
    Args:
      size (int): size of the attention embedding.
    """
    super(NeighbourMultiHeadAttention, self).__init__(has_scatter=True)
    query_size = query_size if query_size is not None else in_size
    self.query_size = query_size
    self.attention_size = attention_size
    self.heads = heads
    self.query = normalization(nn.Linear(query_size, heads * attention_size))
    self.key = normalization(nn.Linear(in_size, heads * attention_size))
    self.value = normalization(nn.Linear(in_size, heads * attention_size))
    self.output = normalization(nn.Linear(heads * attention_size, out_size))

  def attend(self, query, data):
    raise NotImplementedError("Abstract.")

  def reduce_scatter(self, own_data, source_message, indices, node_count):
    target = self.query(own_data).view(*own_data.shape[:-1], -1, self.heads)
    source = self.key(source_message).view(*source_message.shape[:-1], -1, self.heads)
    value = self.value(source_message).view(*source_message.shape[:-1], -1, self.heads)
    attention = scatter.softmax(
      self.attend(target, source), indices, dim_size=node_count
    )
    result = scatter.add(
      (attention.unsqueeze(-2) * value), indices, dim_size=node_count
    )
    result = self.output(result.view(*result.shape[:-2], -1))
    return result

  def reduce(self, own_data, source_message):
    target = self.query(own_data).view(*own_data.shape[:-1], -1, self.heads)
    source = flatten_message(source_message)
    inputs = self.value(source).view(*source_message.shape[:-1], -1, self.heads)
    source = self.key(source).view(*source_message.shape[:-1], -1, self.heads)
    attention = func.softmax(self.attend(target.unsqueeze(1), source), dim=1).unsqueeze(-2)
    out = (attention * inputs).sum(dim=1)
    out = out.view(*out.shape[:-2], -1)
    result = self.output(out)
    return result

class NeighbourDotMultiHeadAttention(NeighbourMultiHeadAttention):
  def attend(self, query, data):
    scaling = torch.sqrt(torch.tensor(self.attention_size, dtype=torch.float))
    return (query * data).sum(dim=-2) / scaling

class NeighbourAddMultiHeadAttention(NeighbourMultiHeadAttention):
  def attend(self, query, data):
    return (query + data).sum(dim=-2)

class NeighbourReducer(ConnectedModule):
  def __init__(self, reduction):
    super(NeighbourReducer, self).__init__()
    self.reduction = reduction

  def reduce(self, own_data, source_message):
    return self.reduction(source_message, dim=1)

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
    self.linear = nn.Linear(2 * channels, channels)

  def forward(self, graph, structure):
    out = self.aggregate(graph, graph, structure)
    out = self.linear(out)
    out = self.activation(out + graph)
    return out
