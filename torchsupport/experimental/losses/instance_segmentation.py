import torch
import torch.nn as nn

class InstanceClusterLoss(nn.Module):
  def __init__(self, var=1.0, dist=1.0, reg=1.0,
               distance_to_center=0.1,
               distance_between_centers=0.2):
    super(InstanceClusterLoss, self).__init__()
    self.var = var
    self.dist = dist
    self.reg = reg
    self.dtc = distance_to_center
    self.dbc = distance_between_centers

  def _var_loss(self, prediction, target):
    loss_val = torch.tensor(0.0)
    mean_embeddings = []
    for polygon in target:
      polygon_values = prediction[polygon]
      mean_embedding = polygon_values.mean()
      polygon_loss = max(torch.norm(polygon_values - mean_embedding) - self.dtc, 0.0) ** 2
      polygon_loss /= polygon.sum()
      loss_val += polygon_loss
      mean_embeddings.append(mean_embedding)
    loss_val /= len(target)
    return loss_val, mean_embeddings

  def _dist_loss(self, mean_embeddings):
    losses = [
      max(2 * self.dbc - torch.norm(e_a - e_b), 0) ** 2
      for ida, e_a in enumerate(mean_embeddings)
      for idb, e_b in enumerate(mean_embeddings)
      if ida != idb
    ]
    N = len(mean_embeddings)
    return sum(losses) / (N * (N - 1))

  def _reg_loss(self, mean_embeddings):
    return sum(map(torch.norm, mean_embeddings)) / len(mean_embeddings)

  def forward(self, prediction, target):
    var_loss, mean_embeddings = self._var_loss(prediction, target)
    reg_loss = self._reg_loss(mean_embeddings)
    dist_loss = self._dist_loss(mean_embeddings)
    return self.var * var_loss + self.dist * dist_loss + self.reg * reg_loss
