import torch
import torch.nn as nn
import torch.nn.functional as func

import torchsupport.modules.reduction as red
from torchsupport.ops.shape import flatten

class MetricLoss(nn.Module):
  """Loss designed to train a true metric, as opposed to a
  sigmoid classifier.
  """
  def __init__(self):
    super(MetricLoss, self).__init__()

  def forward(self, input, target):
    weight = (1.0 - target)
    weight /= weight.sum()
    weight += target / target.sum()
    tensor_result = weight * (input - target) ** 2
    return tensor_result.sum()

class BinaryMetricNetwork(nn.Module):
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
    super(BinaryMetricNetwork, self).__init__()
    self.embedding = embedding
    self.metric = metric
    self.task_embedding = task_embedding
    self.task_representation = None
    self.frozen = False

  def embed_task(self, task):
    """Embeds a single task, and freezes the network for inference.
    """
    self.task_representation = self.task_embedding(task)
    self.frozen = True

  def forward(self, input, task=None):
    if task != None and not self.frozen:
      self.task_representation = self.task_embedding(task)
    input_representation = self.embedding(input)

    result = self.metric(input_representation,
                         self.task_representation)
    result = func.sigmoid(result)
    
    return result

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
    self.task_representation = self.task_embedding(task)
    self.frozen = True

  def forward(self, input, task=None):
    if task != None and not self.frozen:
      self.task_representation = self.task_embedding(task)
    input_representation = self.embedding(input)

    result = torch.zeros((input_representation.size()[0], self.task_representation.size()[0]))
    result = result.to(input.device)

    for idx in range(self.task_representation.size()[0]):
      subtask = self.task_representation[idx, :]
      subresult = flatten(self.metric(input_representation, subtask))
      result[:, idx] = func.sigmoid(subresult)
    
    return result

class ConvexMetricNetwork(MetricNetwork):
  # TODO
  pass

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
    return super(PrototypicalMetricNetwork, self).forward(input, task)

class PrototypicalBinaryMetricNetwork(MetricNetwork):
  def __init__(self, embedding, metric):
    """Learns a representation of data, together with a metric relating two datapoints.

    Arguments
    ---------
    embedding : an embedding for both tasks and network inputs.
    metric : a distance function between an embedded input and an embedded task.
    """
    super(PrototypicalBinaryMetricNetwork, self).__init__(
      embedding,
      red.TaskBinaryPrototype(embedding),
      metric
    )

  def forward(self, input, task=None):
    return super(PrototypicalBinaryMetricNetwork, self).forward(input, task)

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
    return super(ReductionMetricNetwork, self).forward(input, task)

class StatefulReductionMetricNetwork(MetricNetwork):
  def __init__(self, embedding, reduction, metric):
    """Learns a representation of data, tasks and a metric relating datapoints and tasks.

    Arguments
    ---------
    embedding : an embedding for tasks and network inputs.
    reduction : a reduction function compressing multiple datapoints into one.
    metric : a distance function between an embeded input and an embedded task.
    """
    super(StatefulReductionMetricNetwork, self).__init__(
      embedding,
      red.StatefulTaskReduction(embedding, reduction),
      metric
    )

  def forward(self, input, task=None):
    return super(StatefulReductionMetricNetwork, self).forward(input, task)
