import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.gradient import replace_gradient
from torchsupport.structured import scatter
from torchsupport.ops.shape import deshape, reshape

class VQ(nn.Module):
  def __init__(self, in_size, n_vectors, discount=0.9):
    super().__init__()
    self.discount = discount
    self.n_vectors = n_vectors
    self.weights = torch.ones(in_size, dtype=torch.float)
    self.prototypes = torch.randn(n_vectors, in_size, requires_grad=True)

  def update(self, inputs, code):
    # update counts
    unique, count = code.unique(return_counts=True)
    self.weights = self.discount * self.weights
    self.weights[unique] += (1 - self.discount) * count.float()

    # update vectors
    values, indices = code.sort()
    self.prototypes = self.discount * self.prototypes
    scatter.add((1 - self.discount) * inputs[indices], values, out=self.prototypes)
    self.prototypes = self.prototypes / self.weights

  def forward(self, inputs):
    inputs, shape = deshape(inputs)
    distance = (inputs[:, None, :] - self.prototypes[None, :, :]).norm(dim=-1)
    code = distance.argmax(dim=1)
    closest = self.prototypes[code]

    with torch.no_grad():
      self.update(inputs, code)

    one_hot = torch.zeros_like(distance)
    one_hot[torch.arange(code.size(0), device=code.device), code] = 1
    one_hot = replace_gradient(one_hot, inputs)
    one_hot = reshape(one_hot, shape)
    closest = reshape(closest, shape)
    return one_hot, closest
