import torch
import torch.nn as nn
import torch.nn.functional as func

import modules.dynamic as dyn
import modules.compact as com

class MetricNetwork(nn.Module):
  def __init__(self, embedding, metric, squash=None):
    """Learns a representation of data, together with a metric relating two datapoints."""
    self.embedding = embedding
    self.metric = metric
    if squash != None:
      self.support_embedding = squash
    else:
      self.support_embedding = lambda x: x

  def forward(self, input, support):
    support_representation = self.support_embedding(self.embedding(support))
    input_representation = self.embedding(input)

    result = torch.Variable(torch.Tensor(support_representation.size()[0]))

    for idx in range(support_representation.size()[0]):
      element = support_representation[idx, :]
      result[idx] = self.metric(input_representation, support_representation)
    
    return result

# class WeightedReduction(nn.Module):
#   def __init__(self, filters):
#     """Learns a reduction operation on a variably sized set of inputs."""
#     self.filters = filters

#   def forward(self, input):
#     input_shape = input.size()
#     result = input_shape
#     if len(input_shape) == 4:
#       dim = 1
#     else:
#       dim = 0
#     torch.mean(result, dim=dim)