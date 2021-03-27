from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import OneHotCategorical

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
               num_workers=8,
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
    self.num_workers = num_workers

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

  def visualize(self, data):
    pass

  def contrastive_step(self, data):
    """Performs a single step of contrastive training.

    Args:
      data: data points used for training.
    """
    if self.step_id % self.report_interval == 0:
      self.visualize(data)

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
        self.data, batch_size=self.batch_size, num_workers=self.num_workers,
        shuffle=True, drop_last=True
      )

      for data in self.train_data:
        self.step(data)
        self.log()
        self.step_id += 1

    return self.get_netlist(self.names)

class MoCoTraining(AbstractContrastiveTraining):
  def __init__(self, net, data, temperature=0.1, buffer_size=2048,
               momentum=0.99, shuffle=True, **kwargs):
    self.net = ...
    super().__init__({
      "net": net
    }, data, **kwargs)
    self.buffer_size = buffer_size
    self.temperature = temperature
    self.buffer = None
    self.index = None
    self.target = deepcopy(self.net)
    self.momentum = momentum
    self.requires_shuffle = shuffle

  def similarity(self, x, y):
    numerator = (x * y).sum(dim=-1)
    nx = x.norm(dim=-1)
    ny = y.norm(dim=-1)
    denominator = nx * ny
    return numerator / (denominator + 1e-6)

  def update_buffer(self, data):
    self.buffer[self.index:self.index + self.batch_size] = data
    self.index += self.batch_size
    self.index = self.index % self.buffer_size

  def init_buffer(self):
    diter = iter(self.train_data)
    for _ in range(self.buffer_size // self.batch_size):
      data = to_device(next(diter), self.device)
      with torch.no_grad():
        key = self.target(data[0]).cpu()
      if self.buffer is None:
        self.buffer = torch.zeros(self.buffer_size, key.size(1))
        self.index = 0
      self.update_buffer(key)

  def momentum_update(self, target, source):
    with torch.no_grad():
      for t, s in zip(target.parameters(), source.parameters()):
        t *= self.momentum
        t += (1 - self.momentum) * s

  def shuffle(self, data):
    if self.requires_shuffle:
      index = torch.randperm(self.batch_size, device=data.device)
      data = data[index]
      return data, index
    return data, None

  def unshuffle(self, data, index):
    if self.requires_shuffle:
      reindex = torch.arange(self.batch_size, dtype=torch.long, device=data.device)
      reindex[index] = reindex.clone()
      return data[reindex]
    return data

  def run_networks(self, data):
    if self.buffer is None:
      self.init_buffer()
    key, query = data
    query = self.net(query)
    with torch.no_grad():
      self.momentum_update(self.target, self.net)
      key, index = self.shuffle(key)
      key = self.target(key)
      key = self.unshuffle(key, index)
      negatives = to_device(self.buffer, self.device)
      self.update_buffer(key.cpu())
    positive = self.similarity(query, key)
    negative = self.similarity(query[:, None, :], negatives[None, :, :])
    return positive, negative

  def contrastive_loss(self, positive, negative):
    total = torch.cat((positive[:, None], negative), dim=1)
    total = total / self.temperature
    return -total.log_softmax(dim=1)[:, 0].mean()

class EnergyMoCoTraining(MoCoTraining):
  def __init__(self, net, data, alpha=0.3, positive_decay=0.1,
               negative_decay=0.1, **kwargs):
    super().__init__(net, data, **kwargs)
    self.alpha = alpha
    self.positive_decay = positive_decay
    self.negative_decay = negative_decay

  def contrastive_loss(self, positive, negative):
    contrastive = positive.mean() - self.alpha * negative.mean()
    regularization = self.positive_decay * (positive ** 2).mean()
    regularization += self.negative_decay * (negative ** 2).mean()
    return contrastive + regularization

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

class SimSiamTraining(AbstractContrastiveTraining):
  def __init__(self, net, predictor, data, momentum=None, **kwargs):
    r"""Trains a network in a self-supervised manner following the method outlined
    in "Exploring Simple Siamese Representation Learning".

    Args:
      net (nn.Module): network to be trained.
      predictor (nn.Module): predictor to transform representations from different
        data views.
      data (Dataset): dataset to perform training on.
    """
    self.net = ...
    self.predictor = ...
    super().__init__({
      "net": net,
      "predictor": predictor
    }, data, **kwargs)
    self.momentum = momentum
    if self.momentum is not None:
      self.target = deepcopy(self.net)
      self.target = self.target.eval()

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
    sim = sim * (1 - torch.eye(sim.size(0), device=sim.device)[:, :, None])
    size = sim.size(0)
    sim = sim.view(-1, x.size(1)).sum(dim=0) / (size * (size - 1))
    return sim

  def visualize(self, data):
    self.writer.add_images("variant 1", data[0], self.step_id)
    self.writer.add_images("variant 2", data[1], self.step_id)

  def contrastive_step(self, data):
    super().contrastive_step(data)
    if self.momentum is not None:
      with torch.no_grad():
        for parameter, target in zip(self.target.parameters(), self.net.parameters()):
          parameter *= self.momentum
          parameter += (1 - self.momentum) * target

  def run_networks(self, data):
    data = list(map(lambda x: x.unsqueeze(0), data))
    inputs = torch.cat(data, dim=0).view(-1, *data[0].shape[2:])
    features = self.net(inputs)
    predictions = self.predictor(features)
    if self.momentum is not None:
      with torch.no_grad():
        features = self.target(inputs)
    shape = features.shape[1:]
    features = features.view(len(data), -1, *shape)
    predictions = predictions.view(len(data), -1, *shape)
    return features.detach(), predictions

  def contrastive_loss(self, features, predictions):
    result = -self.similarity(predictions, features).mean()
    features = predictions
    norm_features = features
    norm_features = features / features.norm(dim=2, keepdim=True)
    norm_features = norm_features.view(-1, features.size(2)).std(dim=0).mean()
    self.current_losses["std"] = float(norm_features)
    self.current_losses["contrastive"] = float(result)
    return result

class SelfClassifierTraining(SimSiamTraining):
  def __init__(self, net, data, **kwargs):
    super().__init__(net, nn.Identity(), data, **kwargs)

  def run_networks(self, data):
    data = list(map(lambda x: x.unsqueeze(0), data))
    inputs = torch.cat(data, dim=0).view(-1, *data[0].shape[2:])
    features = self.net(inputs)
    shape = features.shape[1:]
    features = features.view(len(data), -1, *shape)
    return (features,)

  def contrastive_loss(self, features):
    log_yx = features.log_softmax(dim=-1)
    p_xy = features.softmax(dim=1)
    value = (p_xy[None, :] * log_yx[:, None]).mean(dim=-1).sum(dim=2)
    value = value * (1 - torch.eye(value.size(0), device=value.device))
    value = value.sum(dim=(0, 1)) / (value.size(0) * (value.size(0) - 1))
    return -value.mean()

class TwinTraining(SelfClassifierTraining):
  redundancy_weight = 5e-3
  def contrastive_loss(self, features):
    mean = features.mean(dim=1, keepdim=True)
    std = features.std(dim=1, keepdim=True)
    features = (features - mean) / std
    correlation = torch.einsum("ixj,kxl->ikjl", features, features) / features.size(1)
    diag = torch.eye(correlation.size(-1), device=correlation.device)
    diag = diag[None, None, :, :]
    corr = ((correlation - diag) ** 2)
    corr = corr * (diag + self.redundancy_weight * (1 - diag))
    loss = corr.sum(dim=(2, 3))
    loss = loss * (1 - torch.eye(loss.size(0), device=loss.device))
    return loss.mean()

class ClassifierSimSiamTraining(SimSiamTraining):
  def __init__(self, *args, hard=False, **kwargs):
    r"""Trains a network in a self-supervised manner following the method outlined
    in "Exploring Simple Siamese Representation Learning" using cross entropy loss
    in a pseudo labelling pretext task.

    Args:
      net (nn.Module): network to be trained.
      predictor (nn.Module): predictor to transform representations from different
        data views.
      data (Dataset): dataset to perform training on.
      hard (bool): use hard pseudolabels?
    """
    super().__init__(*args, **kwargs)
    self.hard = hard

  def similarity(self, x, y):
    logits = x.log_softmax(dim=2)
    if self.hard:
      cat = OneHotCategorical(logits=y).sample()
    else:
      cat = y.softmax(dim=2)
    sim = (logits[None, :] * cat[:, None]).view(x.size(0), x.size(0), x.size(1), -1)
    sim = sim.sum(dim=-1)
    ind = torch.arange(sim.size(0), dtype=torch.long, device=sim.device)
    sim[ind, ind] = 0.0
    count = sim.size(0) ** 2 - sim.size(0)
    sim = sim.view(-1, x.size(1)).sum(dim=0) / count
    return sim
