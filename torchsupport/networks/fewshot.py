import torch
import torch.nn as nn
import torch.nn.functional as func

import modules.dynamic as dyn
import modules.compact as com

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

  def forward(self, input, task):
    task_representation = self.task_embedding(self.embedding(task))
    input_representation = self.embedding(input)

    result = torch.Variable(torch.Tensor(task_representation.size()[0]))

    for idx in range(task_representation.size()[0]):
      element = task_representation[idx, :]
      result[idx] = self.metric(input_representation, task_representation)
    
    return result

class ConvexMetricNetwork(MetricNetwork):
  # TODO

class PrototypeTaskEmbedding(nn.Module):
  def __init__(self, embedding):
    """Embeds a task according to a given combination heuristic.

    Arguments
    ---------
    embedding : an input embedding function.
    """
    super(PrototypeTaskEmbedding, self).__init__()
    self.embedding = embedding

  def forward(self, task):
    inputs = task[0]
    labels = task[1]
    length = len(inputs)

    unique_labels = []
    for label in labels:
      if label not in unique_labels:
        unique_labels.append(label)
    num_labels = len(unique_labels)

    input_representation = self.embedding(inputs)
    result = torch.Variable(torch.zeros(num_labels))

    for idx, label in enumerate(unique_labels):
      mask = labels == label
      result[idx] = torch.sum(input_representation[labels == label], 0)
      result[idx] /= sum(mask).float()

    return result

class ReductionTaskEmbedding(nn.Module):
  def __init__(self, embedding, reduction):
    """Embeds a variable number of labelled support examples by a trainable
    reduction function.
    
    Arguments
    ---------
    embedding : a support example embedding function.
    reduction : a trainable function compacting multiple support examples into
                a single task representation by reduction, generalizing prototypical
                networks.
    """
    super(ReductionTaskEmbedding, self).__init__()
    self.embedding = embedding
    self.reduction = reduction

  def forward(self, task):
    unique_labels = []
    for label in labels:
      if label not in unique_labels:
        unique_labels.append(label)
    num_labels = len(unique_labels)

    input_representation = self.embedding(inputs)
    result = torch.Variable(torch.zeros(num_labels))

    for idx, label in enumerate(unique_labels):
      mask = labels == label
      label_tensor = input_representation[mask]
      for idy in range(label_tensor.size()[0]):
        result[idx] = self.reduction(result[idx], label_tensor[idy])
    
    return result

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
      PrototypeTaskEmbedding(embedding),
      metric
    )

  def forward(self, input, task):
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
      ReductionTaskEmbedding(embedding, reduction),
      metric
    )

  def forward(self, input, task):
    super(ReductionMetricNetwork, self).forward(input, task)
