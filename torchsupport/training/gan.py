
import numpy as np
import torch
from torch import nn
from torch.nn import functional as func
from torch.utils.data import DataLoader
from torch.distributions import Normal, RelaxedOneHotCategorical

from tensorboardX import SummaryWriter

from torchsupport.training.training import Training
import torchsupport.modules.losses.vae as vl
from torchsupport.data.io import netwrite

class AbstractGANTraining(Training):
  """Abstract base class for GAN training."""
  def __init__(self, generators, discriminators, data,
               optimizer=torch.optim.Adam,
               generator_optimizer_kwargs=None,
               discriminator_optimizer_kwargs=None,
               n_critic=1,
               n_actor=1,
               max_epochs=50,
               batch_size=128,
               device="cpu",
               network_name="network",
               verbose=False):
    super(AbstractGANTraining, self).__init__()

    self.verbose = verbose
    self.checkpoint_path = network_name

    self.n_critic = n_critic
    self.n_actor = n_actor

    generator_netlist = []
    self.generator_names = []
    for network in generators:
      self.generator_names.append(network)
      network_object = generators[network]
      setattr(self, network, network_object)
      generator_netlist.extend(list(network_object.parameters()))

    discriminator_netlist = []
    self.discriminator_names = []
    for network in discriminators:
      self.discriminator_names.append(network)
      network_object = discriminators[network]
      setattr(self, network, network_object)
      discriminator_netlist.extend(list(network_object.parameters()))

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

    if generator_optimizer_kwargs is None:
      generator_optimizer_kwargs = {"lr" : 5e-4}
    if discriminator_optimizer_kwargs is None:
      discriminator_optimizer_kwargs = {"lr" : 5e-4}

    self.generator_optimizer = optimizer(
      generator_netlist,
      **generator_optimizer_kwargs
    )
    self.discriminator_optimizer = optimizer(
      generator_netlist,
      **discriminator_optimizer_kwargs
    )

  def generator_loss(self, *args):
    """Abstract method. Computes the generator loss."""
    raise NotImplementedError("Abstract")

  def discriminator_loss(self, *args):
    """Abstract method. Computes the discriminator loss."""
    raise NotImplementedError("Abstract")

  def sample(self, *args, **kwargs):
    """Abstract method. Samples from the latent distribution."""
    raise NotImplementedError("Abstract")

  def run_generator(self, data):
    """Abstract method. Runs generation at each step."""
    raise NotImplementedError("Abstract")

  def run_discriminator(self, data):
    """Abstract method. Runs discriminator training."""
    raise NotImplementedError("Abstract")

  def discriminator_step(self, data):
    self.discriminator_optimizer.zero_grad()
    if isinstance(data, (list, tuple)):
      data = [
        point.to(self.device)
        for point in data
      ]
    elif isinstance(data, dict):
      data = {
        key : data[key].to(self.device)
        for key in data
      }
    else:
      data = data.to(self.device)
    args = self.run_discriminator(data)
    loss_val = self.discriminator_loss(*args)

    if self.verbose:
      for loss_name in self.current_losses:
        loss_float = self.current_losses[loss_float]
        self.writer.add_scalar(f"{loss_name} loss", loss_float, self.step_id)
    self.writer.add_scalar("discriminator total loss", float(loss_val), self.step_id)

    loss_val.backward()
    self.discriminator_optimizer.step()

  def generator_step(self, data):
    self.generator_optimizer.zero_grad()
    if isinstance(data, (list, tuple)):
      data = [
        point.to(self.device)
        for point in data
      ]
    elif isinstance(data, dict):
      data = {
        key : data[key].to(self.device)
        for key in data
      }
    else:
      data = data.to(self.device)
    args = self.run_generator(data)
    loss_val = self.generator_loss(*args)

    if self.verbose:
      for loss_name in self.current_losses:
        loss_float = self.current_losses[loss_float]
        self.writer.add_scalar(f"{loss_name} loss", loss_float, self.step_id)
    self.writer.add_scalar("generator total loss", float(loss_val), self.step_id)

    loss_val.backward()
    self.generator_optimizer.step()

  def step(self, data):
    for _ in range(self.n_critic):
      self.discriminator_step(data)
    for _ in range(self.n_actor):
      self.generator_step(data)
    self.each_step()

  def checkpoint(self):
    for name in self.generator_names:
      netwrite(
        getattr(self, name),
        f"{self.checkpoint_path}-{name}-epoch-{self.epoch_id}-step-{self.step_id}.torch"
      )
    for name in self.discriminator_names:
      netwrite(
        getattr(self, name),
        f"{self.checkpoint_path}-{name}-epoch-{self.epoch_id}-step-{self.step_id}.torch"
      )
    self.each_checkpoint()

  def train(self):
    for epoch_id in range(self.max_epochs):
      self.epoch_id = epoch_id
      self.train_data = None
      self.train_data = DataLoader(
        self.data, batch_size=self.batch_size, num_workers=8,
        shuffle=True
      )
      for data, *_ in self.train_data:
        self.step(data)
        self.step_id += 1
      self.checkpoint()

    generators = [
      getattr(self, name)
      for name in self.generator_names
    ]

    discriminators = [
      getattr(self, name)
      for name in self.discriminator_names
    ]

    return generators, discriminators

class GANTraining(AbstractGANTraining):
  def __init__(self, generator, discriminator, data, **kwargs):
    self.generator = ...
    self.discriminator = ...
    super(GANTraining, self).__init__(
      {"generator": generator},
      {"discriminator": discriminator},
      data, **kwargs
    )

  def sample(self):
    return self.generator.sample(self.batch_size())

  def generator_loss(self, generated):
    loss_val = func.binary_cross_entropy_with_logits(
      self.discriminator(generated), 0
    )

    return loss_val

  def discriminator_loss(self, generated, real,
                         generated_result, real_result):
    generated_loss = func.binary_cross_entropy_with_logits(
      generated_result, 1
    )
    real_loss = func.binary_cross_entropy_with_logits(
      real_result, 0
    )

    return generated_loss + real_loss

  def run_generator(self, data):
    sample = self.sample()
    generated = self.generator(sample)
    return generated

  def run_discriminator(self, data):
    fake_batch = self.run_generator(data)
    fake_result = self.discriminator(fake_batch)
    real_result = self.discriminator(data)
    return data, fake_batch, fake_result, real_result

def _mix_on_path(real, fake):
  sample = torch.rand(
    real.size(0),
    *[1 for _ in range(real.dim() - 1)]
  )
  return real * sample + fake * (1 - sample)

class WGANTraining(GANTraining):
  def __init__(self, generator, discriminator, data, penalty=10, **kwargs):
    super(WGANTraining, self).__init__(generator, discriminator, data, **kwargs)
    self.penalty = penalty

  def discriminator_loss(self, fake, real, generated_result, real_result):
    loss_val = torch.mean(real_result - generated_result)
    penalty_function = \
      (torch.norm(self.discriminator(_mix_on_path(real, fake)), 2) - 1) ** 2
    gradient_penalty = torch.autograd.grad(
      penalty_function,
      self.discriminator.parameters(),
      create_graph=True
    )

    self.current_losses["discriminator"] = float(loss_val)
    self.current_losses["gradient-penalty"] = float(gradient_penalty)

    return loss_val + self.penalty * gradient_penalty

  def generator_loss(self, generated):
    return -self.discriminator(generated)

class GPGANTraining(GANTraining):
  def __init__(self, generator, discriminator, data, penalty=10, **kwargs):
    super(GPGANTraining, self).__init__(generator, discriminator, data, **kwargs)
    self.penalty = penalty

  def discriminator_loss(self, fake, real, generated_result, real_result):
    loss_val = torch.mean(real_result - generated_result)
    penalty_function = self.discriminator(_mix_on_path(real, fake)) ** 2
    gradient_penalty = torch.autograd.grad(
      penalty_function,
      self.discriminator.parameters(),
      create_graph=True
    )

    self.current_losses["discriminator"] = float(loss_val)
    self.current_losses["gradient-penalty"] = float(gradient_penalty)

    return loss_val + self.penalty * gradient_penalty
