import torch


def mean_squared_distance_matrix(x, y=None):
  """
  Computes a mean squared distance matrix
  """
  if y is None:
    y = x
  if x.shape[1] != y.shape[1]:
    raise ValueError(f'Dimensions to compare mismatch (shape[1]): {x.shape} {y.shape}')
  tiled_x = x.unsqueeze(1)
  tiled_y = y.unsqueeze(0)
  return torch.mean((tiled_x - tiled_y) ** 2, dim=2)


def sum_of_squared_distance_matrix(x, y=None):
  """
  Computes a mean squared distance matrix
  """
  if y is None:
    y = x
  if x.shape[1] != y.shape[1]:
    raise ValueError(f'Dimensions to compare mismatch (shape[1]): {x.shape} {y.shape}')
  tiled_x = x.unsqueeze(1)
  tiled_y = y.unsqueeze(0)
  return torch.sum((tiled_x - tiled_y) ** 2, dim=2)


def rbf_distance_matrix(x, y=None, gamma=1.):
  """
  Takes two dim-2 tensors and creates the gaussian RBF kernel distance matrix
  Args:
      x: Tensor(N, C)
      y: Tensor(N, C)
      gamma: rbf hyperparameter
  Returns:
      Distance matrix: Tensor(N_x, N_y)
  """
  if y is None:
    y = x
  return torch.exp(- gamma * sum_of_squared_distance_matrix(x, y))


def multi_rbf_distance_matrix(x, y=None):
  if y is None:
    y = x
  # TODO: remove the ugly hardcode from trVAE paper
  sigmas = torch.tensor([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25,
                         30, 35, 100, 1e3, 1e4, 1e5, 1e6], 
                         device=x.device, requires_grad=False)
  beta = 1 / (2 * sigmas)

  distances = sum_of_squared_distance_matrix(x, y)
  s = distances.unsqueeze(0) * beta.view(-1, *(1 for _ in distances.shape))

  return torch.sum(torch.exp(-s), dim=0) / len(sigmas)
