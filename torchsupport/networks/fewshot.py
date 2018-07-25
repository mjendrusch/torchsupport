import torch
import torch.nn as nn
import torch.nn.functional as func

import torchsupport.modules.dynamic as dyn
import torchsupport.modules.compact as com
import torchsupport.modules.reduction as red

class MetricNetwork(nn.Module):
  def __init__(self, embedding, task_embedding, metric):
    """Learns a representation of data, together with a metric relating two datapoints.
    
    Arguments
    ---------
    embedding : An embedding function for network inputs.
    metric : a distance function between an embedded input and an embedded task.
    task_embedding : An embedding function for network tasks. A Task is represented by
                     a pair of Datastructures, one containing the input, the other
                     containing its annotation.
    """
    super(MetricNetwork, self).__init__()
    self.embedding = embedding
    self.metric = metric
    self.task_embedding = task_embedding
    self.task_representation = None
    self.frozen = False

  def embed_task(self, task):
    """Embeds a single task, and freezes the network for inference.
    """
    self.task_representation = self.task_embedding(self.embedding(task))
    self.frozen = True

  def forward(self, input, task=None):
    if task != None and not self.frozen:
      self.task_representation = self.task_embedding(self.embedding(task))
    input_representation = self.embedding(input)

    result = torch.Variable(torch.Tensor(self.task_representation.size()[0]))

    for idx in range(self.task_representation.size()[0]):
      element = self.task_representation[idx, :]
      result[idx] = self.metric(input_representation, self.task_representation)
    
    return result

class ConvexMetricNetwork(MetricNetwork):
  # TODO

class PrototypicalMetricNetwork(MetricNetwork):
  def __init__(self, embedding, metric):
    """Learns a representation of data, together with a metric relating two datapoints.

    Arguments
    ---------
    embedding : an embedding for both tasks and network inputs.
    metric : a distance function between an embedded input and an embedded task.
    """
    super(PrototypicalMetricNetwork, self).__init__(
      embedding,
      red.TaskPrototype(embedding),
      metric
    )

  def forward(self, input, task=None):
    super(PrototypicalMetricNetwork, self).forward(input, task)

class ReductionMetricNetwork(MetricNetwork):
  def __init__(self, embedding, reduction, metric):
    """Learns a representation of data, tasks and a metric relating datapoints and tasks.

    Arguments
    ---------
    embedding : an embedding for tasks and network inputs.
    reduction : a reduction function compressing multiple datapoints into one.
    metric : a distance function between an embeded input and an embedded task.
    """
    super(ReductionMetricNetwork, self).__init__(
      embedding,
      red.TaskReduction(embedding, reduction),
      metric
    )

  def forward(self, input, task=None):
    super(ReductionMetricNetwork, self).forward(input, task)
