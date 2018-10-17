import torch
import torch.nn as nn
import torch.nn.functional as func
from torchsupport.ops.shape import batchexpand, flatten

class Combination(nn.Module):
  def __init__(self, combinator, evaluator):
    """Structural element for disjoint combination / evaluation networks.

    Arguments
    ---------
    combinator : a network joining two disjoint input tensors, for example
                 by concatenation.
    evaluator : a network taking a combined tensor, and performing
                computation on that tensor.
    """
    super(Combination, self).__init__()
    self.combinator = combinator
    self.evaluator = evaluator

  def forward(self, input, task):
    combination = self.combinator(input, task)
    result = self.evaluator(combination)
    return result

class Concatenation(Combination):
    def __init__(self, evaluator):
        """Structural element concatenating two tensors.

        Arguments
        ---------
        evaluator : a network taking the concatenated tensor, and
                    performing computation on that tensor.
        """
        super(Concatenation, self).__init__(
            lambda input, task: _concatenate(input, task),
            evaluator
        )
    
    def forward(self, input, task):
        return super(Concatenation, self).forward(input, task)

def _concatenate(input, task):
    flattened_input = flatten(input, batch=True)
    flattened_task = flatten(task, batch=False)
    concatenated = torch.cat([
      flattened_input.unsqueeze(1),
      batchexpand(flattened_task, flattened_input).unsqueeze(1)
    ], 1)
    print("CONCSHAPE: ", concatenated.size())
    return concatenated

class ConnectedCombination(Concatenation):
    def __init__(self, evaluator, inputs, outputs, bn=True):
        """Structural element performing a linear map on a concatenation
        of two Tensors.
        
        Arguments
        ---------
        evaluator : a network taking a combined tensor, and
                    performing computation on that tensor.
        inputs : the number of input features.
        outputs : the desired number of output features.
        bn : if True, perform batch normalization.
        """
        super(ConnectedCombination, self).__init__(evaluator)
        self.connected = nn.Linear(inputs, outputs)
        if bn:
            self.bn = nn.BatchNorm1d(outputs)
        else:
            self.bn = None

    def forward(self, input, task):
        concatenated = _concatenate(input, task)
        combined = self.connected(concatenated)
        combined = func.dropout(combined, training=True)
        if self.bn != None:
            combined = self.bn(flatten(combined, batch=True)).unsqueeze(1)
        result = self.evaluator(combined)
        return result

class BilinearCombination(Combination):
    def __init__(self, evaluator, inputs, outputs, bn=True):
        """Structural element combining two tensors by bilinear transformation.

        Arguments
        ---------
        evaluator : a network taking a combined tensor, and
                    performing computation on that tensor.
        inputs : an array or tuple `[inputs1, inputs2]` containing the number
                 of input features for each input tensor.
        otuputs : the desired number of output features.
        bn : if True, perform batch normalization.
        """
        bilinear = nn.Bilinear(*inputs, outputs)
        if bn:
            bn_layer = nn.BatchNorm1d(outputs)
        else:
            bn_layer = None
        super(BilinearCombination, self).__init__(
            lambda input, task: self.compute(input, task),
            evaluator
        )
        self.bilinear = bilinear
        self.bn = bn_layer

    def compute(self, input, task):
        flattened_input = flatten(input, batch=True)
        flattened_task = flatten(task, batch=False)
        flattened_task = batchexpand(flattened_task, flattened_input)
        result = self.bilinear(flattened_input, flattened_task)
        if self.bn != None:
            result = self.bn(result)
        return result

    def forward(self, input, task):
        return super(BilinearCombination, self).forward(input, task)