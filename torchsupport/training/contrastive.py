from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.training.state import (
  NetNameListState, TrainingState
)
from torchsupport.training.training import Training
from torchsupport.data.io import to_device, make_differentiable
from torchsupport.data.collate import DataLoader, default_collate

class AbstractContrastiveTraining(Training):
  """Abstract base class for contrastive training."""
  checkpoint_parameters = Training.checkpoint_parameters + [
    TrainingState(),
    NetNameListState("names")
  ]
  def __init__(self, networks, data,
               optimizer=torch.optim.Adam,
               optimizer_kwargs=None,
               **kwargs):
    """Generic training setup for energy/score based models.

    Args:
      networks (list): networks used for contrastive learning.
      data (Dataset): provider of training data.
      optimizer (Optimizer): optimizer class for gradient descent.
      optimizer_kwargs (dict): keyword arguments for the
        optimizer used in score function training.
    """
    super().__init__(**kwargs)

    netlist = []
    self.names, netlist = self.collect_netlist(networks)

    self.data = data
    self.train_data = None

    self.current_losses = {}

    if optimizer_kwargs is None:
      optimizer_kwargs = {"lr" : 5e-4}

    self.optimizer = optimizer(
      netlist,
      **optimizer_kwargs
    )

    self.checkpoint_names = self.get_netlist(self.names)

  def contrastive_loss(self, *args):
    """Abstract method. Computes the contrastive loss."""
    raise NotImplementedError("Abstract")

  def regularization(self, *args):
    """Computes network regularization."""
    return 0.0

  def loss(self, *args):
    contrastive = self.contrastive_loss(*args)
    regularization = self.regularization(*args)
    return contrastive + regularization

  def run_networks(self, data):
    """Abstract method. Runs networks at each step."""
    raise NotImplementedError("Abstract")

  def contrastive_step(self, data):
    """Performs a single step of contrastive training.

    Args:
      data: data points used for training.
    """
    self.optimizer.zero_grad()
    data = to_device(data, self.device)
    make_differentiable(data)
    args = self.run_networks(data)
    loss_val = self.loss(*args)

    self.log_statistics(loss_val, name="total loss")

    loss_val.backward()
    self.optimizer.step()

  def step(self, data):
    """Performs a single step of contrastive training.

    Args:
      data: data points used for training.
    """
    self.contrastive_step(data)
    self.each_step()

  def train(self):
    """Runs contrastive training until the maximum number of epochs is reached."""
    for epoch_id in range(self.max_epochs):
      self.epoch_id = epoch_id
      self.train_data = None
      self.train_data = DataLoader(
        self.data, batch_size=self.batch_size, num_workers=8,
        shuffle=True, drop_last=True
      )

      for data in self.train_data:
        self.step(data)
        self.log()
        self.step_id += 1

    return self.get_netlist(self.names)

class SimCLRTraining(AbstractContrastiveTraining):
  def __init__(self, net, data, temperature=0.1, **kwargs):
    r"""Trains a network in a self-supervised manner following the method outlined
    in "A Simple Framework for Contrastive Learning of Visual Representations".

    Args:
      net (nn.Module): network to be trained.
      data (Dataset): dataset to perform training on.
      temperature (float): softmax temperature parameter.
    """
    self.net = ...
    super().__init__({
      "net": net
    }, data, **kwargs)
    self.temperature = temperature

  def similarity(self, x):
    r"""Computes the mutual similarity between a batch of data representations.

    Args:
      x (torch.Tensor): batch of data to compute similarities of.
    """
    numerator = (x[:, None] * x[None, :]).sum(dim=-1)
    denominator = x[:, None].norm(dim=-1) * x[None, :].norm(dim=-1)
    return numerator / (denominator + 1e-6)

  def run_networks(self, data):
    latent = self.net(torch.cat(data, dim=0))
    return latent

  def contrastive_loss(self, latent):
    size = latent.size(0) // 2
    sim = self.similarity(latent)
    max_sim = sim.max()
    vals = ((sim - max_sim) / self.temperature)
    exp = vals.exp()
    ind = torch.arange(exp.size(0))
    ind_shift = (ind + size) % size
    log_numerator = vals[ind, ind_shift]
    log_denominator = (exp.sum(dim=1) - exp[ind, ind]).log
    result = (-log_numerator + log_denominator).mean()
    self.current_losses["contrastive"] = float(result)
    return result

class ScoreSimCLRTraining(SimCLRTraining):
  def __init__(self, net, data, score_matching_scale=1.0, **kwargs):
    r"""Trains a network in a self-supervised manner using SimCLR, regularized
    to minimize a score-matching loss. Inherits arguments from :class:`SimCLRTraining`.

    Args:
      net (nn.Module): network to be trained.
      data (Dataset): dataset to perform training on.
      score_matching_scale (float): scaling parameter for the score matching regularization.
    """
    super().__init__(net, data, **kwargs)
    self.score_matching_scale = score_matching_scale

  def run_networks(self, data):
    x, y = data
    v_x = torch.randn_like(x)
    v_y = torch.randn_like(y)
    x_p = x + v_x
    y_p = y + v_y
    x_m = x - v_x
    y_m = y - v_y
    plain = super().run_networks((x, y))
    plus = super().run_networks((x_p, y_p))
    minus = super().run_networks((x_m, y_m))
    return plain, plus, minus

  def contrastive_loss(self, plain, plus, minus):
    return super().contrastive_loss(plain)

  def regularization(self, plain, plus, minus):
    plain = plain.logsumexp(dim=1)
    plus = plus.logsumexp(dim=1)
    minus = minus.logsumexp(dim=1)
    result = plus + minus - 2 * plain
    result = result + ((plus - minus) ** 2) / 8
    result = result.mean()
    self.current_losses["score matching"] = result
    return self.score_matching_scale * result

class BYOLTraining(AbstractContrastiveTraining):
  def __init__(self, net, predictor, data, averaging_rate=0.99, **kwargs):
    r"""Trains a network in a self-supervised manner following the method outlined
    in "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning".

    Args:
      net (nn.Module): network to be trained.
      data (Dataset): dataset to perform training on.
      averaging_rate (float): rate of exponential averaging of the target network.
    """
    self.net = ...
    self.predictor = ...
    super().__init__({
      "net": net,
      "predictor": predictor
    }, data, **kwargs)
    self.target = deepcopy(self.net)
    self.tau = averaging_rate

  def similarity(self, x, y):
    result = (x * y).sum(dim=1)
    result = result / (x.norm(dim=1) * y.norm(dim=1))
    return result

  def run_networks(self, data):
    data = torch.cat(data, dim=0)
    latent = self.net(data)
    prediction = self.predictor(latent)
    target_latent = self.target(data).chunk(2, dim=0)
    target_latent = torch.cat(tuple(reversed(target_latent)), dim=0)
    return prediction, target_latent

  def contrastive_loss(self, prediction, target_latent):
    sim = self.similarity(prediction, target_latent)
    result = 2 * sim.mean()
    self.current_losses["contrastive"] = float(result)
    return result

  def contrastive_step(self, data):
    super().contrastive_step(data)
    with torch.no_grad():
      for parameter, target in zip(self.target.parameters(), self.net.parameters()):
        parameter *= self.tau
        parameter += (1 - self.tau) * target

class BYOCTraining(BYOLTraining):
  def __init__(self, net, data, **kwargs):
    r"""Trains a network in a self-supervised manner using a modification to
    BYOL, where the network is trained to match classifications returned by a
    target networks.

    Args:
      net (nn.Module): network to be trained.
      data (Dataset): dataset to perform training on.
      averaging_rate (float): rate of exponential averaging of the target network.
    """
    super().__init__(net, nn.Identity(), data, **kwargs)

  def similarity(self, x, y):
    target = y.argmax(dim=1)
    return func.cross_entropy(x, target, reduction='none')

class ScoreBYOCTraining(BYOCTraining):
  def __init__(self, net, data, score_matching_scale=1.0,
               finite_difference_eps=1e-1, **kwargs):
    r"""Trains a network in a self-supervised manner using a modification to
    BYOL, where the network is trained to match classifications returned by a
    target networks.

    Args:
      net (nn.Module): network to be trained.
      data (Dataset): dataset to perform training on.
      averaging_rate (float): rate of exponential averaging of the target network.
    """
    super().__init__(net, nn.Identity(), data, **kwargs)
    self.score_matching_scale = score_matching_scale
    self.finite_difference_eps = finite_difference_eps

  def run_networks(self, data):
    data = torch.cat(data, dim=0)
    noise = torch.rand_like(data)
    noise = noise / noise.norm(
      dim=tuple(range(1, noise.dim)),
      keepdim=True
    )
    noise = noise * self.finite_difference_eps

    plus = data + noise
    minus = data - noise

    inputs = torch.cat((data, plus, minus), dim=0)

    latent, plus, minus = self.net(inputs).chunk(3, dim=0)
    target_latent = self.target(data).chunk(2, dim=0)
    target_latent = torch.cat(tuple(reversed(target_latent)), dim=0)
    return latent, plus, minus, target_latent

  def contrastive_loss(self, prediction, plus, minus, target_latent):
    return super().contrastive_loss(prediction, target_latent)

  def regularization(self, plain, plus, minus, target_latent):
    plain = plain.logsumexp(dim=1)
    plus = plus.logsumexp(dim=1)
    minus = minus.logsumexp(dim=1)
    result = plus + minus - 2 * plain
    result = result + ((plus - minus) ** 2) / 8
    result = result.mean()
    self.current_losses["score matching"] = float(result)
    return self.score_matching_scale * result
