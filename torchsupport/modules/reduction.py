import torch
import torch.nn as nn
import torch.nn.functional as func

class Prototype(nn.Module):
    # TODO
    def __init__(self, embedding, indices, dim):
        """Embeds a task according to a given combination heuristic.

        Arguments
        ---------
        embedding : an input embedding function.
        """
        super(Prototype, self).__init__()
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

class Reduction(nn.Module):
    # TODO
    def __init__(self, embedding, reduction, indices, dim):
        """Embeds a variable number of labelled support examples by a trainable
        reduction function.
        
        Arguments
        ---------
        embedding : a support example embedding function.
        reduction : a trainable function compacting multiple support examples into
                    a single task representation by reduction, generalizing prototypical
                    networks.
        """
        super(Reduction, self).__init__()
        self.embedding = embedding
        self.reduction = reduction

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
        label_tensor = input_representation[mask]
        for idy in range(label_tensor.size()[0]):
            result[idx] = self.reduction(result[idx], label_tensor[idy])
        
        return result

class TaskPrototype(nn.Module):
    def __init__(self, embedding):
        """Embeds a task according to a given combination heuristic.

        Arguments
        ---------
        embedding : an input embedding function.
        """
        super(Prototype, self).__init__()
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

class TaskReduction(nn.Module):
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
        super(Reduction, self).__init__()
        self.embedding = embedding
        self.reduction = reduction

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
        label_tensor = input_representation[mask]
        for idy in range(label_tensor.size()[0]):
            result[idx] = self.reduction(result[idx], label_tensor[idy])
        
        return result


class StatefulTaskReduction(nn.Module):
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
        super(Reduction, self).__init__()
        self.embedding = embedding
        self.reduction = reduction

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
        label_tensor = input_representation[mask]
        state = self.reduction.initState()
        for idy in range(label_tensor.size()[0]):
            state, result[idx] = self.reduction(state, label_tensor[idy])
        
        return result