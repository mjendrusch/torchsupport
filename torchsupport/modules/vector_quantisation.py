import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.gradient import replace_gradient
from torchsupport.structured import scatter
from torchsupport.ops.shape import deshape, reshape

class VQ(nn.Module):
  r"""Quantises a batch of input vectors to a learnable set of
  discrete symbols.

  Shape:
    - Inputs: :math:`(N, C_{in}, ...)`
    - Outputs: :math:`(N, N_{vectors}, ...)`

  Args:
    in_size (int): number of input features.
    n_vectors (int): number of output symbols.
    discount (float): factor of exponential averaging for
      prototype parameters. Default: 0.9

  .. warning::
      in its current iteration, the vector quantisation module
      does not yet support shared-memory or distributed data
      parallel execution. Each copy of the module would maintain
      its own copy of prototype parameters.
  """
  def __init__(self, in_size, n_vectors, discount=0.9):
    super().__init__()
    self.discount = discount
    self.n_vectors = n_vectors
    self.weights = torch.ones(in_size, dtype=torch.float)
    self.combinations = torch.randn(n_vectors, in_size, requires_grad=True)
    self.prototypes = self.combinations / self.weights

  def update(self, inputs, code):
    r"""Updates prototype representations given a batch of inputs.

    Args:
      inputs (torch.Tensor): batch of continuous inputs.
      code (torch.Tensor): batch of symbol assignments.

    .. warning::
        currently, this update is not synchronised across
        multiple threads or processes holding copies of this
        module, resulting in independent copies of symbol
        prototypes.
    """
    # update counts
    unique, count = code.unique(return_counts=True)
    self.weights = self.discount * self.weights
    self.weights[unique] += (1 - self.discount) * count.float()

    # update vectors
    values, indices = code.sort()
    self.combinations = self.discount * self.combinations
    scatter.add((1 - self.discount) * inputs[indices], values, out=self.combinations)
    self.prototypes = self.combinations / self.weights

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

class VectorQuantization(nn.Module):
  def __init__(self, size, code_size=512, beta=0.25):
    super().__init__()
    self.beta = beta
    self.codebook = nn.Embedding(code_size, size)
    with torch.no_grad():
      self.codebook.weight.uniform_(-1 / code_size, 1 / code_size)

  def forward(self, inputs):
    shape = inputs.shape
    out = inputs.reshape(*shape[:2], -1).permute(2, 0, 1).reshape(-1, shape[1])
    dist = ((out[:, None, :] - self.codebook.weight[None, :, :]) ** 2).sum(dim=-1)
    indices = dist.argmin(dim=1)
    closest = self.codebook(indices)
    assignment = ((closest.detach() - out) ** 2).mean()
    shift = ((out.detach() - closest) ** 2).mean()
    loss = assignment + self.beta * shift
    straight_through = replace_gradient(closest, out)
    straight_through = straight_through.reshape(-1, shape[0], shape[1])
    straight_through = straight_through.permute(1, 2, 0)
    straight_through = straight_through.view(*shape)
    indices = indices.reshape(-1, shape[0]).permute(1, 0).reshape(shape[0], *shape[2:])
    return straight_through, indices, loss
