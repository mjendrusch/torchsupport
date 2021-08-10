import random
from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.autograd as ag
from torch.utils.data import Dataset

from tensorboardX import SummaryWriter

from torchsupport.training.state import (
  State, NetState, NetNameListState, TrainingState, PathState
)
from torchsupport.training.samplers import Langevin, AnnealedLangevin
from torchsupport.training.training import Training
from torchsupport.data.io import netwrite, to_device, make_differentiable, detach
from torchsupport.data.collate import DataLoader, default_collate
from torchsupport.modules.losses.vae import normal_kl_loss

class AbstractEnergyTraining(Training):
  """Abstract base class for GAN training."""
  checkpoint_parameters = Training.checkpoint_parameters + [
    TrainingState(),
    NetNameListState("names")
  ]
  def __init__(self, scores, data,
               optimizer=torch.optim.Adam,
               optimizer_kwargs=None,
               num_workers=8,
               **kwargs):
    """Generic training setup for energy/score based models.

    Args:
      scores (list): networks used for scoring.
      data (Dataset): provider of training data.
      optimizer (Optimizer): optimizer class for gradient descent.
      optimizer_kwargs (dict): keyword arguments for the
        optimizer used in score function training.
      max_epochs (int): maximum number of training epochs.
      batch_size (int): number of training samples per batch.
      device (string): device to use for training.
      network_name (string): identifier of the network architecture.
      verbose (bool): log all events and losses?
    """
    super(AbstractEnergyTraining, self).__init__(**kwargs)

    netlist = []
    self.names = []
    for network in scores:
      self.names.append(network)
      network_object = scores[network].to(self.device)
      setattr(self, network, network_object)
      netlist.extend(list(network_object.parameters()))

    self.num_workers = num_workers
    self.data = data
    self.train_data = None

    self.current_losses = {}

    if optimizer_kwargs is None:
      optimizer_kwargs = {"lr" : 5e-4}

    self.optimizer = optimizer(
      netlist,
      **optimizer_kwargs
    )

    self.checkpoint_names = {
      name: getattr(self, name)
      for name in self.names
    }

  def energy_loss(self, *args):
    """Abstract method. Computes the score function loss."""
    raise NotImplementedError("Abstract")

  def loss(self, *args):
    return self.energy_loss(*args)

  def prepare(self, *args, **kwargs):
    """Abstract method. Prepares an initial state for sampling."""
    raise NotImplementedError("Abstract")

  def sample(self, *args, **kwargs):
    """Abstract method. Samples from the Boltzmann distribution."""
    raise NotImplementedError("Abstract")

  def run_energy(self, data):
    """Abstract method. Runs score at each step."""
    raise NotImplementedError("Abstract")

  def each_generate(self, *inputs):
    """Reports on generation."""
    pass

  def energy_step(self, data):
    """Performs a single step of discriminator training.

    Args:
      data: data points used for training.
    """
    self.optimizer.zero_grad()
    data = to_device(data, self.device)
    args = self.run_energy(data)
    loss_val = self.loss(*args)

    if self.verbose:
      for loss_name in self.current_losses:
        loss_float = self.current_losses[loss_name]
        self.writer.add_scalar(f"{loss_name} loss", loss_float, self.step_id)
    self.writer.add_scalar("discriminator total loss", float(loss_val), self.step_id)

    loss_val.backward()
    self.optimizer.step()

  def step(self, data):
    """Performs a single step of GAN training, comprised of
    one or more steps of discriminator and generator training.

    Args:
      data: data points used for training."""
    self.energy_step(data)
    self.each_step()

  def train(self):
    """Trains an EBM until the maximum number of epochs is reached."""
    for epoch_id in range(self.max_epochs):
      self.train_data = None
      self.train_data = DataLoader(
        self.data, batch_size=self.batch_size, num_workers=self.num_workers,
        shuffle=True, drop_last=True
      )

      for data in self.train_data:
        self.step(data)
        self.log()
        self.step_id += 1
      self.epoch_id += 1

    scores = [
      getattr(self, name)
      for name in self.names
    ]

    return scores

class SampleBuffer(Dataset):
  def __init__(self, owner, buffer_size=10000,
               buffer_probability=0.95,
               accept_probability=1.0):
    self.samples = []
    self.current_size = 0
    self.owner = owner
    self.current = 0
    self.seen = 0
    self.buffer_size = buffer_size
    self.buffer_probability = buffer_probability
    self.accept_probability = accept_probability

  def reset(self):
    self.samples = []

  def random_position(self):
    return random.randrange(len(self.samples))

  def probability(self):
    if isinstance(self.accept_probability, float):
      return self.accept_probability
    return self.accept_probability(self.buffer_size, self.seen)

  def update(self, results):
    for result in results:
      if len(self.samples) < self.buffer_size:
        self.samples.append(result)
      else:
        flip = random.random()
        if flip <= self.probability():
          self.samples[self.random_position()] = result
      self.seen += 1
      self.current = (self.current + 1) % self.buffer_size

  def sample(self, count):
    result = [
      self.samples[random.randrange(len(self.samples))]
      for idx in range(count)
    ]
    result = default_collate(result)
    return result

  def __getitem__(self, idx):
    initialized = len(self.samples) > self.owner.batch_size
    if random.random() < self.buffer_probability and initialized:
      result = self.samples[idx % len(self.samples)]
    else:
      result = self.owner.prepare()
    return result

  def __len__(self):
    return self.buffer_size

class EnergyTraining(AbstractEnergyTraining):
  checkpoint_parameters = AbstractEnergyTraining.checkpoint_parameters + [
    PathState(["buffer", "samples"])
  ]
  def __init__(self, score, *args, buffer_size=10000, buffer_probability=0.95,
               sample_steps=10, decay=1, reset_threshold=1000, integrator=None,
               oos_penalty=True, accept_probability=1.0, sampler_likelihood=1.0,
               maximum_entropy=0.3, **kwargs):
    self.score = ...
    super(EnergyTraining, self).__init__(
      {"score": score}, *args, **kwargs
    )
    self.sampler_likelihood = sampler_likelihood
    self.maximum_entropy = maximum_entropy
    self.target_score = deepcopy(score).eval()
    self.reset_threshold = reset_threshold
    self.oos_penalty = oos_penalty
    self.decay = decay
    self.integrator = integrator if integrator is not None else Langevin()
    self.sample_steps = sample_steps
    self.buffer = SampleBuffer(
      self, buffer_size=buffer_size,
      buffer_probability=buffer_probability,
      accept_probability=accept_probability
    )
    self.buffer_loader = lambda x: DataLoader(
      x, batch_size=self.batch_size, shuffle=True, drop_last=True
    )

  def update_target(self):
    with torch.no_grad():
      for (target, source) in zip(
          self.target_score.parameters(),
          self.score.parameters()
      ):
        target.set_(source)

  def data_key(self, data):
    result, *args = data
    return (result, *args)

  def each_step(self):
    if abs(self.current_losses["ebm"]) > self.reset_threshold:
      self.reset()
    super().each_step()

  def reset(self):
    self.load()
    self.buffer.reset()

  def each_generate(self, data, *args):
    pass

  def decompose_batch(self, data, *args):
    if isinstance(data, (list, tuple)):
      return (
        to_device(([item[idx] for item in data], *[arg[idx] for arg in args]), "cpu")
        for idx in range(len(data[0]))
      )
    return (
      to_device((data[idx], *[arg[idx] for arg in args]), "cpu")
      for idx in range(len(data))
    )

  def sample(self):
    buffer_iter = iter(self.buffer_loader(self.buffer))
    data, *args = to_device(self.data_key(next(buffer_iter)), self.device)
    self.score.eval()
    data = detach(self.integrator.integrate(self.score, data, *args))
    self.score.train()
    detached = to_device(data, "cpu")
    args = detach(args)
    update = self.decompose_batch(detached, *args)
    self.buffer.update(update)

    return to_device((detached, *args), self.device)

  def compare(self, source, target):
    result = 0.0
    if isinstance(source, (list, tuple)):
      for s_part, t_part in zip(source, target):
        result += self.compare(s_part, t_part)
    else:
      ind = torch.arange(
        source.size(0), dtype=torch.long,
        device=source.device
      )
      diff = (source[:, None] - target[None, :])
      diff = diff.reshape(*diff.shape[:2], -1)
      diff = diff.norm(dim=-1)
      diff[ind, ind] = diff.max().detach()
      result = diff.min(dim=1)[0]
    return result

  def energy_loss(self, real_result, fake_result, fake_update_result, comparison):
    regularization = self.decay * ((real_result ** 2).mean() + (fake_result ** 2).mean())
    ebm = real_result.mean() - fake_result.mean()

    opt = 0.0
    if self.sampler_likelihood:
      opt = fake_update_result.mean()
      self.current_losses["opt"] = float(opt)

    ent = 0.0
    if self.maximum_entropy:
      ent = comparison.log().mean()
      self.current_losses["ent"] = float(ent)

    self.current_losses["real"] = float(real_result.mean())
    self.current_losses["fake"] = float(fake_result.mean())
    self.current_losses["regularization"] = float(regularization)
    self.current_losses["ebm"] = float(ebm)
    return regularization + ebm + self.sampler_likelihood * opt - self.maximum_entropy * ent

  def run_energy(self, data):
    make_differentiable(data)
    input_data, *data_args = self.data_key(data)
    real_result = self.score(input_data, *data_args)

    # sample after first pass over real data, to catch
    # possible batch-norm shenanigans without blowing up.
    fake = self.sample()

    if self.step_id % self.report_interval == 0:
      detached, *args = self.data_key(to_device(fake, "cpu"))
      self.each_generate(detach(detached), *args)

    make_differentiable(fake)
    input_fake, *fake_args = self.data_key(fake)

    self.score.eval()
    fake_update_result = None
    if self.sampler_likelihood:
      # Sampler log likelihood:
      fake_result = self.score(input_fake, *fake_args)
      fake_update = self.integrator.step(self.score, input_fake, *fake_args)
      self.update_target()
      fake_update_result = self.target_score(fake_update, *fake_args)

    comparison = None
    if self.maximum_entropy:
      # Sampler entropy:
      compare_update = fake_update
      compare_target = to_device(self.data_key(self.buffer.sample(100))[0], compare_update.device)
      if hasattr(self.score, "embed"):
        compare_update = self.target_score.embedding(compare_update)
        compare_target = self.target_score.embedding(compare_target)
      comparison = self.compare(compare_update, compare_target.detach())
    self.score.train()
    return real_result, fake_result, fake_update_result, comparison

class SetVAETraining(EnergyTraining):
  def divergence_loss(self, parameters):
    return normal_kl_loss(*parameters)

  def loss(self, real_result, fake_result, oos_result, real_parameters, fake_parameters, oos_parameters):
    energy_loss = self.energy_loss(real_result, fake_result)
    if self.oos_penalty:
      energy_loss -= (oos_result).mean()
    kld_loss = (self.divergence_loss(fake_parameters) + self.divergence_loss(real_parameters)) / 2

    self.current_losses["energy"] = float(energy_loss)
    self.current_losses["kullback leibler"] = float(kld_loss)

    return energy_loss + 1e-4 * kld_loss

  def prepare(self):
    data = self.data[random.randrange(len(self.data))]
    _, reference, *condition = self.data_key(data)
    return (torch.rand_like(reference), reference, *condition)

  def bad_prepare(self):
    _, reference, *args = self.prepare()
    _, bad_reference, *_ = self.prepare()
    noise = 0.1 * torch.rand(bad_reference.size(0), 1, 1, 1)
    bad_reference = (1 - noise) * bad_reference + noise * torch.rand_like(bad_reference)
    result = (bad_reference, reference, *args)
    return result

  def out_of_sample(self):
    result = []
    for idx in range(self.batch_size):
      sample = self.bad_prepare()
      result.append(sample)
    data, *args = to_device(default_collate(result), self.device)
    #data = self.integrator.integrate(self.score, data, *args).detach()
    detached = to_device(data.detach(), "cpu")

    return to_device((data, *args), self.device)

  def run_energy(self, data):
    fake = self.sample()
    if self.oos_penalty:
      oos = self.out_of_sample()

    if self.step_id % self.report_interval == 0:
      detached, *args = self.data_key(fake)
      self.each_generate(detached, *args)

    make_differentiable(fake)
    make_differentiable(data)
    if self.oos_penalty:
      make_differentiable(oos)
    input_data, reference_data, *data_args = self.data_key(data)
    input_fake, reference_fake, *fake_args = self.data_key(fake)
    if self.oos_penalty:
      input_oos, reference_oos, *oos_args = self.data_key(oos)
    real_result, real_parameters = self.score(input_data, reference_data, *data_args)
    fake_result, fake_parameters = self.score(input_fake, reference_fake, *fake_args)
    oos_result = None
    oos_parameters = None
    if self.oos_penalty:
      oos_result, oos_parameters = self.score(input_oos, reference_oos, *oos_args)
    return real_result, fake_result, oos_result, real_parameters, fake_parameters, oos_parameters

class CyclicSetVAETraining(SetVAETraining):
  def loss(self, real_result, fake_result, oos_result, real_parameters, fake_parameters, oos_parameters):
    energy_loss = self.energy_loss(real_result, fake_result)
    if self.oos_penalty:
      energy_loss -= (oos_result).mean()
    kld_loss = (self.divergence_loss(fake_parameters) + self.divergence_loss(real_parameters)) / 2

    self.current_losses["energy"] = float(energy_loss)
    self.current_losses["kullback leibler"] = float(kld_loss)

    scale = (self.step_id % 1000) / 1000
    scale = 1.0 if self.step_id % 2000 >= 1000 else scale

    return energy_loss + scale * kld_loss

class DenoisingScoreTraining(EnergyTraining):
  def __init__(self, score, data, *args, sigma=1, factor=0.60, n_sigma=10, **kwargs):
    self.score = ...
    EnergyTraining.__init__(
      self, score, data, *args,
      buffer_size=1, buffer_probability=0.0,
      integrator=None, **kwargs
    )

    self.sigma = sigma
    self.factor = factor
    self.n_sigma = n_sigma

  def data_key(self, data):
    result, *args = data
    return (result, *args)

  def each_generate(self, data, *args):
    pass

  def noise(self, data):
    scale = torch.randint(0, self.n_sigma, (data.size(0),))
    sigma = self.sigma * self.factor ** scale.float()
    sigma = sigma.to(self.device)
    sigma = sigma.view(*sigma.shape, *((data.dim() - sigma.dim()) * [1]))
    noise = data + sigma * torch.randn_like(data)
    return noise, sigma

  def prepare_sample(self):
    results = []
    for idx in range(self.batch_size):
      results.append(self.prepare())
    return default_collate(results)

  def sample(self):
    self.score.eval()
    with torch.no_grad():
      integrator = AnnealedLangevin([
        self.sigma * self.factor ** idx for idx in range(self.n_sigma)
      ])
      prep = to_device(self.prepare_sample(), self.device)
      data, *args = self.data_key(prep)
      result = integrator.integrate(self.score, data, *args).detach()
    self.score.train()
    return to_device((result, data, *args), self.device)

  def energy_loss(self, score, data, noisy, sigma):
    raw_loss = 0.5 * sigma ** 2 * ((score + (noisy - data) / sigma ** 2) ** 2)
    raw_loss = raw_loss.sum(dim=1, keepdim=True)
    self.current_losses["ebm"] = float(raw_loss.mean())
    return raw_loss.mean()

  def each_step(self):
    AbstractEnergyTraining.each_step(self)
    if self.step_id % self.report_interval == 0 and self.step_id != 0:
      data, *args = self.sample()
      self.each_generate(data, *args)

  def run_energy(self, data):
    data, *args = self.data_key(data)
    noisy, sigma = self.noise(data)
    make_differentiable(noisy)
    result = self.score(noisy, sigma, *args)
    return (
      result,#.view(result.size(0), -1),
      data,#.view(result.size(0), -1),
      noisy,#.view(result.size(0), -1),
      sigma,#.view(result.size(0), -1)
    )

class SlicedScoreTraining(DenoisingScoreTraining):
  def noise_vectors(self, score):
    return torch.randn_like(score)

  def energy_loss(self, score, data, noisy, sigma):
    vectors = self.noise_vectors(score)
    grad_v = (score * vectors).view(score.size(0), -1).sum()
    jacobian = ag.grad(grad_v, noisy, create_graph=True)[0]

    norm = (score ** 2).view(score.size(0), -1).sum(dim=-1) / 2
    jacobian = (vectors * jacobian).view(score.size(0), -1).sum(dim=-1)

    result = (norm + jacobian) * sigma.view(score.size(0), -1) ** 2
    result = result.mean()

    self.current_losses["ebm"] = float(result)

    return result

class MultiscaleScoreTraining(DenoisingScoreTraining):
  def __init__(self, score, data, *args, sigma_0=0.1, **kwargs):
    super().__init__(score, data, *args, **kwargs)
    self.sigma_0 = sigma_0

  def run_energy(self, data):
    data, *args = self.data_key(data)
    noisy, sigma = self.noise(data)
    result = self.score(noisy, *args)
    return (
      result.view(result.size(0), -1),
      data.view(result.size(0), -1),
      noisy.view(result.size(0), -1),
      sigma.view(result.size(0), -1)
    )

  def energy_loss(self, score, data, noisy, sigma):
    raw_loss = 0.5 / (sigma ** 2) * ((self.sigma_0 ** 2 * score + (noisy - data)) ** 2)
    raw_loss = raw_loss.sum(dim=1, keepdim=True)
    self.current_losses["ebm"] = float(raw_loss.mean())
    return raw_loss.mean()

class SetScoreVAETraining(DenoisingScoreTraining):
  def divergence_loss(self, parameters):
    return normal_kl_loss(*parameters)

  def loss(self, score, data, noisy, sigma, parameters):
    energy_loss = self.energy_loss(score, data, noisy, sigma)
    kld_loss = self.divergence_loss(parameters)

    self.current_losses["energy"] = float(energy_loss)
    self.current_losses["kullback leibler"] = float(kld_loss)

    return energy_loss + kld_loss

  def noise(self, data):
    scale = torch.randint(0, self.n_sigma, (data.size(0), data.size(1)))
    sigma = self.sigma * self.factor ** scale.float()
    sigma = sigma.to(self.device)
    sigma = sigma.view(*sigma.shape, *((data.dim() - sigma.dim()) * [1]))
    noise = data + sigma * torch.randn_like(data)
    return noise, sigma

  def prepare(self):
    data = self.data[random.randrange(len(self.data))]
    _, reference, *condition = self.data_key(data)
    return (torch.rand_like(reference), reference, *condition)

  def run_energy(self, data):
    data, reference, *args = self.data_key(data)
    noisy, sigma = self.noise(data)
    result, parameters = self.score(noisy, sigma, reference, *args)
    loss_data = data.view(data.size(0) * data.size(1), -1)
    loss_noisy = noisy.view(noisy.size(0) * noisy.size(1), -1)
    loss_sigma = sigma.view(noisy.size(0) * noisy.size(1), -1)
    loss_score = result.view(noisy.size(0) * noisy.size(1), -1)
    return loss_score, loss_data, loss_noisy, loss_sigma, parameters
