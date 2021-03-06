import torch
import torch.nn as nn
import torch.nn.functional as func

class Prototype(nn.Module):
  def __init__(self, embedding):
    """Embeds a task according to a given combination heuristic.

    Args:
      embedding (nn.Module): input embedding function.
    """
    super(Prototype, self).__init__()
    self.embedding = embedding

  def forward(self, task):
    inputs = task[0]
    labels = task[1]

    unique_labels = []
    for label in labels:
      if label not in unique_labels:
        unique_labels.append(label)
    num_labels = len(unique_labels)

    input_representation = self.embedding(inputs)
    result = torch.zeros(num_labels)

    for idx, label in enumerate(unique_labels):
      mask = (labels == label)[:, 0, 0]
      result[idx] = torch.sum(input_representation[mask], 0)
      result[idx] /= sum(mask).float()

    return result

class Reduction(nn.Module):
  def __init__(self, embedding, reduction):
    """Embeds a variable number of labelled support examples by a trainable
    reduction function.
    
    Args:
      embedding (nn.Module): support example embedding function.
      reduction (nn.Module): trainable function compacting multiple support examples into
                             single task representation by reduction, generalizing prototypical
                             networks.
    """
    super(Reduction, self).__init__()
    self.embedding = embedding
    self.reduction = reduction

  def forward(self, task):
    inputs = task[0]
    labels = task[1]

    unique_labels = []
    for label in labels:
      if label not in unique_labels:
        unique_labels.append(label)
    num_labels = len(unique_labels)

    input_representation = self.embedding(inputs)
    result = torch.zeros(num_labels)

    for idx, label in enumerate(unique_labels):
      mask = (labels == label)[:, 0, 0]
    label_tensor = input_representation[mask]
    for idy in range(label_tensor.size()[0]):
      result[idx] = self.reduction(result[idx], label_tensor[idy])
    
    return result

class TaskPrototype(nn.Module):
  def __init__(self, embedding):
    """Embeds a task according to a given combination heuristic.

    Args:
      embedding (nn.Module): input embedding function.
    """
    super(TaskPrototype, self).__init__()
    self.embedding = embedding

  def forward(self, task):
    inputs = task[0]
    labels = task[1]

    unique_labels = []
    for label in labels:
      if label not in unique_labels:
        unique_labels.append(label)
    num_labels = len(unique_labels)

    input_representation = self.embedding(inputs)
    results = []

    for idx, label in enumerate(unique_labels):
      mask = (labels == label)[:, 0, 0]
      sumval = torch.sum(input_representation[mask], 0) / sum(mask).float().item()
      results.append(sumval.unsqueeze(0))

    result = torch.cat(results, dim=0)
    result = result.to(inputs.device)
    return result

class TaskBinaryPrototype(nn.Module):
  def __init__(self, embedding):
    """Embeds a task according to a given combination heuristic.

    Args:
      embedding (nn.Module): input embedding function.
    """
    super(TaskBinaryPrototype, self).__init__()
    self.embedding = embedding

  def forward(self, task):
    inputs = task[0]
    labels = task[1]

    input_representation = self.embedding(inputs)

    result = torch.sum(input_representation, 0) / float(input_representation.size(0))
    return result

class TaskReduction(nn.Module):
  def __init__(self, embedding, reduction):
    """Embeds a variable number of labelled support examples by a trainable
    reduction function.
    
    Args:
      embedding (nn.Module): support example embedding function.
      reduction (nn.Module): trainable function compacting multiple support examples into
                             a single task representation by reduction, generalizing prototypical
                             networks.
    """
    super(TaskReduction, self).__init__()
    self.embedding = embedding
    self.reduction = reduction

  def forward(self, task):
    inputs = task[0]
    labels = task[1]

    unique_labels = []
    for label in labels:
      if label not in unique_labels:
        unique_labels.append(label)
    num_labels = len(unique_labels)

    input_representation = self.embedding(inputs)
    result = torch.zeros(num_labels, input_representation.size()[1])

    for idx, label in enumerate(unique_labels):
      mask = (labels == label)[:, 0, 0]
    label_tensor = input_representation[mask]
    for idy in range(label_tensor.size()[0]):
      result[idx, :] = self.reduction(result[idx, :], label_tensor[idy])
    
    return result


class StatefulTaskReduction(nn.Module):
  def __init__(self, embedding, reduction):
    """Embeds a variable number of labelled support examples by a trainable
    reduction function.
    
    Args:
    embedding : a support example embedding function.
    reduction : a trainable function compacting multiple support examples into
                a single task representation by reduction, generalizing prototypical
                networks.
    """
    super(StatefulTaskReduction, self).__init__()
    self.embedding = embedding
    self.reduction = reduction

  def forward(self, task):
    inputs = task[0]
    labels = task[1]

    unique_labels = []
    for label in labels:
      if label not in unique_labels:
        unique_labels.append(label)
    num_labels = len(unique_labels)

    input_representation = self.embedding(inputs)
    result = torch.zeros(num_labels)

    for idx, label in enumerate(unique_labels):
      mask = (labels == label)[:, 0, 0]
    label_tensor = input_representation[mask]
    state = self.reduction.initState()
    for idy in range(label_tensor.size()[0]):
      state, result[idx] = self.reduction(state, label_tensor[idy])
    
    return result
