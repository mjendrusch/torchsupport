import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

def kmeans(input, n_clusters=16, tol=1e-6):
  """
  TODO: check correctness
  """
  indices = torch.Tensor(np.random.choice(input.size(-1), n_clusters))
  values = input[:, :, indices]

  while True:
    dist = func.pairwise_distance(
      input.unsqueeze(2).expand(-1, -1, values.size(2), input.size(2)).reshape(
        input.size(0), input.size(1), input.size(2) * values.size(2)),
      values.unsqueeze(3).expand(-1, -1, values.size(2), input.size(2)).reshape(
        input.size(0), input.size(1), input.size(2) * values.size(2))
    )
    choice_cluster = torch.argmin(dist, dim=1)
    old_values = values
    values = input[choice_cluster.nonzeros()]
    shift = (old_values - values).norm(dim=1)
    if shift.max() ** 2 < tol:
      break

  return values


def gaussian_kernel(x, sigma=4):
  """Gaussian distance kernel.

  Args:
    x (Tensor): difference between two input features.
    sigma (float): standard deviation of the gaussian kernel.

  Returns:
    The Gaussian kernel for features `x` and standard deviation `sigma`.
  """
  return torch.exp(- (x ** 2).sum(dim=-1) / sigma)

class FHDF2d(nn.Module):
  def __init__(self, spatial_kernel=gaussian_kernel, feature_kernel=gaussian_kernel, clustering=kmeans):
    """Performs fast high-dimensional filtering using clustering.

    Args:
      spatial_kernel (callable): spatial distance kernel used for filtering.
        Defaults to `gaussian_kernel`.
      feature_kernel (callable): feature distance kernel used for filtering.
        Defaults to `gaussian_kernel`.
      clustering (callable): clustering algorithm used for filter approximation.
        Defaults to `kmeans`.

    Note:
      This computes a _dense_ filter over a 2D image. For a sparse filter, methods based
      on sparse matrix-matrix multiplication could provide better performance.
    """
    super(FHDF2d, self).__init__()
    self.spatial_kernel = spatial_kernel
    self.feature_kernel = feature_kernel
    self.clustering = clustering

  def forward(self, input, guide):
    """
    TODO: check correctness
    """
    padding = (input.size(-2) // 2, input.size(-1) // 2)
    clusters = self.clustering(guide.reshape(guide.size(0), guide.size(1), -1))
    A = torch.Tensor([[self.feature_kernel(k - l) for l in clusters] for k in clusters])
    Ad = A.pinverse()
    bk = self.feature_kernel(clusters - guide)
    ck = Ad.mv(bk)
    omega = self.spatial_kernel(input.size(-2), input.size(-1))
    phi = self.feature_kernel(guide - clusters)
    phi_f = phi * input
    rk = func.conv2d(omega, phi, padding=padding)
    vk = func.conv2d(omega, phi_f, padding=padding)
    eta = (ck * rk).sum(dim=1)
    result = (ck * vk).sum(dim=1) / eta
    return result
