import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.autograd as ag
from torch.utils.data import Dataset

from tensorboardX import SummaryWriter

from torchsupport.training.training import Training
from torchsupport.data.io import netwrite, to_device
from torchsupport.data.collate import DataLoader, default_collate
from torchsupport.modules.losses.vae import normal_kl_loss

class AbstractEnergyTraining(Training):
  """Abstract base class for GAN training."""
  def __init__(self, scores, data,
               optimizer=torch.optim.Adam,
               optimizer_kwargs=None,
               max_epochs=50,
               batch_size=128,
               device="cpu",
               network_name="network",
               verbose=False,
               report_steps=10):
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
    super(AbstractEnergyTraining, self).__init__()

    self.verbose = verbose
    self.report_steps = report_steps
    self.checkpoint_path = network_name

    netlist = []
    self.names = []
    for network in scores:
      self.names.append(network)
      network_object = scores[network].to(device)
      setattr(self, network, network_object)
      netlist.extend(list(network_object.parameters()))

    self.data = data
    self.train_data = None
    self.max_epochs = max_epochs
    self.batch_size = batch_size
    self.device = device

    self.current_losses = {}
    self.network_name = network_name
    self.writer = SummaryWriter(network_name)

    self.epoch_id = 0
    self.step_id = 0

    if optimizer_kwargs is None:
      optimizer_kwargs = {"lr" : 5e-4}
    
    self.optimizer = optimizer(
      netlist,
      **optimizer_kwargs
    )

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

  def checkpoint(self):
    """Performs a checkpoint of all generators and discriminators."""
    for name in self.names:
      the_net = getattr(self, name)
      if isinstance(the_net, torch.nn.DataParallel):
        the_net = the_net.module
      netwrite(
        the_net,
        f"{self.checkpoint_path}-{name}-epoch-{self.epoch_id}-step-{self.step_id}.torch"
      )
    self.each_checkpoint()

  def train(self):
    """Trains an EBM until the maximum number of epochs is reached."""
    for epoch_id in range(self.max_epochs):
      self.epoch_id = epoch_id
      self.train_data = None
      self.train_data = DataLoader(
        self.data, batch_size=self.batch_size, num_workers=8,
        shuffle=True, drop_last=True
      )

      for data in self.train_data:
        self.step(data)
        self.step_id += 1
      self.checkpoint()

    scores = [
      getattr(self, name)
      for name in self.names
    ]

    return scores

def _make_differentiable(data):
  if isinstance(data, torch.Tensor):
    data.requires_grad_(True)
  elif isinstance(data, (list, tuple)):
    for item in data:
      _make_differentiable(item)
  elif isinstance(data, dict):
    for key in data:
      _make_differentiable(data[key])
  else:
    pass

class SampleBuffer(Dataset):
  def __init__(self, owner, buffer_size=10000, buffer_probability=0.95):
    self.samples = [
      owner.prepare()
      for idx in range(buffer_size)
    ]
    self.owner = owner
    self.current = 0
    self.buffer_probability = buffer_probability

  def update(self, results):
    for result in results:
      self.samples[self.current] = result
      self.current = (self.current + 1) % len(self)

  def __getitem__(self, idx):
    if random.random() < self.buffer_probability:
      result = self.samples[(self.current + idx) % len(self)]
    else:
      result = self.owner.prepare()
    return result

  def __len__(self):
    return len(self.samples)

def clip_grad_by_norm(gradient, max_norm=0.01):
  norm = torch.norm(gradient)
  if norm > max_norm:
    gradient = gradient * (max_norm / norm)
  return gradient

class Langevin(nn.Module):
  def __init__(self, rate=100.0, noise=0.005, steps=10, max_norm=0.01):
    super(Langevin, self).__init__()
    self.rate = rate
    self.noise = noise
    self.steps = steps
    self.max_norm = max_norm

  def integrate(self, score, data, *args):
    for idx in range(self.steps):
      _make_differentiable(data)
      _make_differentiable(args)
      energy, *_ = score(data + self.noise * torch.randn_like(data), *args)
      gradient = ag.grad(energy, data, torch.ones(*energy.shape, device=data.device, requires_grad=True))[0]
      if self.max_norm:
        gradient = clip_grad_by_norm(gradient, self.max_norm)
      data = data - self.rate * gradient
      data = data.clamp(0, 1)
      # data = data - self.noise / 2 * gradient + self.noise * torch.randn_like(data)
    return data

class AnnealedLangevin(nn.Module):
  def __init__(self, noises, steps=100, epsilon=2e-5):
    super(AnnealedLangevin, self).__init__()
    self.noises = noises
    self.steps = steps
    self.epsilon = epsilon

  def integrate(self, score, data, *args):
    print("DASH", data.shape)
    for noise in self.noises:
      step_size = self.epsilon * (noise ** 2) / (self.noises[-1] ** 2)
      noise = torch.ones(data.size(0), data.size(1), 1, 1, 1).to(data.device) * noise
      for step in range(self.steps):
        #print(data[0].max(), data[0].min(), data[0].mean())
        _make_differentiable(data)
        _make_differentiable(args)
        gradient = score(data, noise, *args)
        data = data + 0.5 * step_size * gradient + np.sqrt(step_size) * torch.randn_like(data)
        data = data.clamp(0, 1)
    print("GSH", gradient.shape)
    return data

class EnergyTraining(AbstractEnergyTraining):
  def __init__(self, score, *args, buffer_size=100, buffer_probability=0.9,
               sample_steps=10, decay=1, integrator=None, **kwargs):
    self.score = ...
    super(EnergyTraining, self).__init__(
      {"score": score}, *args, **kwargs
    )
    self.decay = decay
    self.integrator = integrator if integrator is not None else Langevin()
    self.sample_steps = sample_steps
    self.buffer = SampleBuffer(
      self, buffer_size=buffer_size, buffer_probability=buffer_probability
    )
    self.buffer_loader = DataLoader(self.buffer, batch_size=self.batch_size, shuffle=False)

  def data_key(self, data):
    result, *args = data
    return (result, *args)

  def each_generate(self, data, *args):
    pass

  def sample(self):
    data, *args = to_device(self.data_key(next(iter(self.buffer_loader))), self.device)
    data = self.integrator.integrate(self.score, data, *args).detach()
    detached = data.detach().cpu()
    update = (to_device((data[idx], *[arg[idx] for arg in args]), "cpu") for idx in range(data.size(0)))
    self.buffer.update(update)

    return to_device((data, *args), self.device)

  def energy_loss(self, real_result, fake_result):
    regularization = (self.decay * (real_result ** 2 + fake_result ** 2)).mean()
    ebm = (real_result - fake_result).mean()
    self.current_losses["real"] = float(real_result.mean())
    self.current_losses["fake"] = float(fake_result.mean())
    self.current_losses["regularization"] = float(regularization)
    self.current_losses["ebm"] = float(ebm)
    return regularization + ebm

  def run_energy(self, data):
    fake = self.sample()

    if self.step_id % 10 == 0:
      detached, *args = self.data_key(fake)
      self.each_generate(detached, *args)

    _make_differentiable(fake)
    _make_differentiable(data)
    input_data, *data_args = self.data_key(data)
    input_fake, *fake_args = self.data_key(fake)
    real_result = self.score(input_data, *data_args)
    fake_result = self.score(input_fake, *fake_args)
    return real_result, fake_result

class SetVAETraining(EnergyTraining):
  def divergence_loss(self, parameters):
    return normal_kl_loss(*parameters)

  def loss(self, real_result, fake_result, oos_result, real_parameters, fake_parameters, oos_parameters):
    print(oos_result.shape, real_result.shape)
    #energy_loss = 0.5 * self.energy_loss(real_result, oos_result)
    energy_loss = self.energy_loss(real_result, fake_result)
    energy_loss -= (oos_result).mean()
    kld_loss = (self.divergence_loss(oos_parameters) + self.divergence_loss(fake_parameters) + self.divergence_loss(real_parameters)) / 3

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
    noise = torch.rand(bad_reference.size(0), 1, 1, 1)
    bad_reference = (1 - noise) * bad_reference + noise * torch.rand_like(bad_reference)
    result = (bad_reference, reference, *args)
    return result

  def out_of_sample(self):
    result = []
    for idx in range(self.batch_size):
      sample = self.bad_prepare()
      result.append(sample)
    data, *args = to_device(default_collate(result), self.device)
    data = self.integrator.integrate(self.score, data, *args).detach()
    detached = data.detach().cpu()

    return to_device((data, *args), self.device)

  def run_energy(self, data):
    fake = self.sample()
    oos = self.out_of_sample()

    if self.step_id % 10 == 0:
      detached, *args = self.data_key(fake)
      self.each_generate(detached, *args)

    _make_differentiable(fake)
    _make_differentiable(data)
    input_data, reference_data, *data_args = self.data_key(data)
    input_fake, reference_fake, *fake_args = self.data_key(fake)
    input_oos, reference_oos, *oos_args = self.data_key(oos)
    real_result, real_parameters = self.score(input_data, reference_data, *data_args)
    fake_result, fake_parameters = self.score(input_fake, reference_fake, *fake_args)
    oos_result, oos_parameters = self.score(input_oos, reference_oos, *oos_args)
    return real_result, fake_result, oos_result, real_parameters, fake_parameters, oos_parameters

class DenoisingScoreTraining(EnergyTraining):
  def __init__(self, score, data, *args, sigma=1, factor=0.8, n_sigma=10, **kwargs):
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
    # noise = data.clamp(0, 1)
    return noise, sigma

  def prepare_sample(self):
    results = []
    for idx in range(self.batch_size):
      results.append(self.prepare())
    return default_collate(results)

  def sample(self):
    integrator = AnnealedLangevin([self.sigma * self.factor ** idx for idx in range(self.n_sigma)], epsilon=2e-4)
    data, *args = self.data_key(self.prepare_sample())
    data = integrator.integrate(self.score, data, *args).detach()
    update = ((data[idx], *[arg[idx] for arg in args]) for idx in range(data.size(0)))
    self.buffer.update(update)

    return to_device((data, *args), self.device)

  def energy_loss(self, score, data, noisy, sigma):
    raw_loss = 0.5 * sigma ** 2 * ((score + (noisy - data) / sigma ** 2) ** 2)
    raw_loss = raw_loss.sum(dim=1, keepdim=True)
    return raw_loss.mean()

  def each_step(self):
    super(DenoisingScoreTraining, self).each_step()
    if self.step_id % 10 == 0:
      data, *args = self.sample()
      self.each_generate(data, *args)

  def run_energy(self, data):
    data, *args = self.data_key(data)
    noisy, sigma = self.noise(data)
    result = self.score(noisy, sigma, *args)
    return result.view(result.size(0), -1), data.view(result.size(0), -1), noisy.view(result.size(0), -1), sigma.view(result.size(0), -1)

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
    print(data.shape, sigma.shape)
    noise = data + sigma * torch.randn_like(data)
    # noise = data.clamp(0, 1)
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
