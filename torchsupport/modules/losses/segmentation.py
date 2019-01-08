import torch
import torch.nn as nn
import torch.nn.functional as func

class CRFLoss(nn.Module):
  def __init__(self, n_classes, radius=2, size=(224, 224)):
    """
    TODO
    """
    pass

class KernelCutLoss(nn.Module):
  def __init__(self, n_classes, radius=2, size=(224, 224), sigma_dist=1, sigma_feat=1):
    """
    TODO
    """
    pass

def intra_class_variance(classes, features):
  """
  Computes the intra-class variance of a tensor of features segmented by
  a tensor of classes.
  Args:
    classes (torch.Tensor): a softmaxed tensor of classes per pixel.
    features (torch.Tensor): a tensor of features whose intra-class variance
                             has to be computed.
  """
  dim_b = classes.size(0)
  dim_x = classes.size(1)
  dim_y = classes.size(2)
  dim_c = classes.size(3) * features.size(3)
  classes = classes.unsqueeze(3).expand(dim_b, dim_x, dim_y, dim_c)
  features = features.unsqueeze(4).expand(dim_b, dim_x, dim_y, dim_c)
  weighted = classes * features
  class_weight = classes.sum(dim=-2)
  mean = weighted.sum(dim=-2) / class_weight
  unbiased = class_weight - (classes ** 2).sum(dim=-2) / class_weight
  variance = ((weighted - mean) ** 2).sum(dim=-2) / unbiased
  return variance

def region_cohesion(classes):
  """
  Computes a measure of region cohesion, that is, the ratio between
  class edges and surface.
  """
  dx = torch.conv2d(classes, torch.tensor([[1, -2, 1]]))
  dy = torch.conv2d(classes, torch.tensor([[1], [-2], [1]]))
  mag = torch.sqrt(dx ** 2 + dy ** 2)
  total_mag = mag.sum(dim=-2)
  total_class = classes.sum(dim=-2)
  return total_mag / total_class

def mask_size(classes):
  """
  Computes per-class segmentation mask size.
  """
  return classes.sum(dim=-2)

class SegmentReconstructionLoss(nn.Module):
  def __init__(self, reconstructor, loss=nn.MSELoss()):
    """
    Reconstruction loss for reconstruction from semantic segmentation labels.
    Args:
      reconstructor (torch.nn.Module): module performing the reconstruction.
      loss (torch.nn.Module): module computing the reconstruction loss.
    """
    self.reconstructor = reconstructor
    self.loss = loss

  def forward(self, classes, target, features=None):
    if features == None:
      reconstruction = self.reconstructor(classes)
    else:
      reconstruction = self.reconstructor(classes, features)
    return self.loss(reconstruction, target)

class NCutLoss(nn.Module):
  def __init__(self, n_classes, radius=2, size=(224, 224), sigma_dist=1, sigma_feat=1):
    """
    Soft normalized cut loss (TODO: requires optimization).
    Args:
      n_classes (int): number of semantic labels.
      radius (int): radius of the gaussian distance kernel.
      size (int, int): size of the input image map.
      sigma_dist (float): standard deviation of the gaussian distance kernel.
      signa_feat (float): standard deviation of the gaussian feature kernel.
    """
    super(NCutLoss, self).__init__()
    self.radius = radius
    self.size = size
    self.n_classes = n_classes
    self.sigma_dist = sigma_dist
    self.sigma_feat = sigma_feat
    self.in_radius_indices = self._in_radius_indices()

  def _in_radius_indices(self):
    """
    TODO: optimize
    """
    result = []
    for idx_0 in range(self.size[0]):
      for idy_0 in range(self.size[1]):
        for idx_1 in range(self.size[0]):
          for idy_1 in range(self.size[1]):
            dist = torch.sqrt((idx_0 - idy_0) ** 2 + (idx_1 - idy_1) ** 2)
            if dist < self.radius:
              result.append(((idx_0, idy_0), (idx_1, idy_1)))
    return result

  def _single_weight(self, features, i, j):
    dist = torch.sqrt((i[0] - j[0]) ** 2 + (i[1] - j[1]) ** 2)
    feat_dist = torch.sqrt(((features[:, :, i] - features[:, :, j]) ** 2).sum(dim=1))
    return torch.exp(
      - dist ** 2 / self.sigma_dist ** 2
      - feat_dist ** 2 / self.sigma_feat ** 2
    )

  def _weight(self, features):
    weights = torch.zeros(self.features.size(0), len(self.in_radius_indices))
    indices = torch.zeros(self.features.size(0), len(self.in_radius_indices), 2)
    for pos, subscript in enumerate(self.in_radius_indices):
      weights[:, pos] = self._single_weight(features, *subscript)
      indices[:, pos, 0] = self._sub_to_ind(subscript[0])
      indices[:, pos, 1] = self._sub_to_ind(subscript[1])
    return torch.sparse_tensor_coo(
      indices,
      weights,
      (self.features.size(0), self.size[0] * self.size[1], self.size[0] * self.size[1])
    )

  def forward(self, classification, features):
    """
    TODO: check correctness
    """
    weights = self._weight(features)
    S = classification
    Sp = torch.transpose(classification, 0, 1, 3, 2)
    degree = torch.transpose(weights.sum(dim=1), 0, 2, 1)
    numerator = Sp.mm(weights.mm(S))
    denominator = degree.mm(S)
    return (numerator / denominator).sum()
