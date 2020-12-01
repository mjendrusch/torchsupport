import torch
import torch.nn as nn

from torchsupport.training.contrastive import SimSiamTraining

class MultiscaleSimSiamTraining(SimSiamTraining):
  def __init__(self, nets, predictors, data, beta=1.0, **kwargs):
    r"""Trains a network in a self-supervised manner following the method outlined
    in "Exploring Simple Siamese Representation Learning".

    Args:
      net (nn.Module): network to be trained.
      predictor (nn.Module): predictor to transform representations from different
        data views.
      data (Dataset): dataset to perform training on.
    """
    super().__init__(nn.ModuleList(nets), nn.ModuleList(predictors), data, **kwargs)
    self.beta = beta

  def similarity(self, x, y):
    r"""Computes the mutual similarity between a batch of data representations.

    Args:
      x (torch.Tensor): batch of data to compute similarities of.
      y (torch.Tensor): batch of data to compute similarities of.
    """
    x = x / x.norm(dim=2, keepdim=True)
    y = y / y.norm(dim=2, keepdim=True)
    sim = (x[None, :] * y[:, None]).view(x.size(0), x.size(0), x.size(1), -1)
    sim = sim.sum(dim=-1)
    sim = sim.view(-1, x.size(1)).mean(dim=0)
    return sim

  def run_networks(self, data):
    global_results = []
    dense_results = []
    policy_results = []
    previous = None
    for (oriented, unoriented, mask), net, predictor in zip(
        data, self.net, self.predictor
    ):
      # unoriented
      unoriented = list(map(lambda x: x.unsqueeze(0), unoriented))
      inputs = torch.cat(unoriented, dim=0).view(-1, *unoriented[0].shape[2:])
      features = net(inputs, predict_map=False)
      predictions = predictor(features)

      # oriented
      oriented_features, dense_features, logits = net(oriented, mask=mask, data=previous)
      oriented_predictions = predictor(oriented_features)
      dense_predictions = dense_features
      previous = oriented_predictions

      shape = features.shape[1:]
      features = features.view(-1, oriented_features.size(0), *shape)
      predictions = predictions.view(-1, oriented_features.size(0), *shape)
      features = torch.cat((features, oriented_features[None]), dim=0)
      predictions = torch.cat((predictions, oriented_predictions[None]), dim=0)

      # save
      global_results.append((features.detach(), predictions))
      dense_results.append((dense_features.detach(), dense_predictions))
      policy_results.append(logits)
    return global_results, dense_results, policy_results

  def contrastive_loss(self, global_results, dense_results, policy_results):
    result = 0.0
    # per level loss
    for idx, (features, predictions) in enumerate(global_results):
      level_loss = super().contrastive_loss(features, predictions)
      result += level_loss
      self.current_losses[f"contrastive level {idx}"] = float(level_loss)

    # inter-level prediction loss
    prediction_losses = []
    for idx, (drop_0, dense_predictions), (features, drop_1) in enumerate(zip(
        dense_results[:-1], global_results[1:]
    )):
      level_loss = -self.similarity(dense_predictions[None], features)
      prediction_losses.append(level_loss.detach())
      level_loss = level_loss.mean()
      result += level_loss
      self.current_losses[f"level prediction {idx + 1}"] = float(level_loss)

    # policy loss
    if self.beta is not None:
      for idx, (logits, reward) in enumerate(zip(policy_results[:-1], prediction_losses)):
        level_loss = logits * (self.beta * (reward - reward.mean())).exp()
        level_loss = level_loss.mean()
        result += level_loss
        self.current_losses[f"level policy loss {idx}"] = float(level_loss)

    self.current_losses["contrastive"] = float(result)
    return result
