
import numpy as np
import torch
from torch import nn
from torch.nn import functional as func
from torch.distributions import Normal, RelaxedOneHotCategorical

from tensorboardX import SummaryWriter

from torchsupport.training.training import Training
import torchsupport.modules.losses.vae as vl
from torchsupport.structured import DataParallel as SDP
from torchsupport.data.io import netwrite, to_device, detach
from torchsupport.data.collate import DataLoader

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
               verbose=False,
               report_steps=10):
    """Generic training setup for generative adversarial networks.

    Args:
      generators (list): networks used in the generation step.
      discriminators (list): networks used in the discriminator step.
      data (Dataset): provider of training data.
      optimizer (Optimizer): optimizer class for gradient descent.
      generator_optimizer_kwargs (dict): keyword arguments for the
        optimizer used in generator training.
      discriminator_optimizer_kwargs (dict): keyword arguments for the
        optimizer used in discriminator training.
      n_critic (int): number of critic training iterations per step.
      n_actor (int): number of actor training iterations per step.
      max_epochs (int): maximum number of training epochs.
      batch_size (int): number of training samples per batch.
      device (string): device to use for training.
      network_name (string): identifier of the network architecture.
      verbose (bool): log all events and losses?
    """
    super(AbstractGANTraining, self).__init__()

    self.verbose = verbose
    self.report_steps = report_steps
    self.checkpoint_path = network_name

    self.n_critic = n_critic
    self.n_actor = n_actor

    generator_netlist = []
    self.generator_names = []
    for network in generators:
      self.generator_names.append(network)
      network_object = generators[network].to(device)
      setattr(self, network, network_object)
      generator_netlist.extend(list(network_object.parameters()))

    discriminator_netlist = []
    self.discriminator_names = []
    for network in discriminators:
      self.discriminator_names.append(network)
      network_object = discriminators[network].to(device)
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
      discriminator_netlist,
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

  def each_generate(self, *inputs):
    """Reports on generation."""
    pass

  def discriminator_step(self, data):
    """Performs a single step of discriminator training.

    Args:
      data: data points used for training.
    """
    self.discriminator_optimizer.zero_grad()
    data = to_device(data, self.device)
    args = self.run_discriminator(data)
    loss_val, *grad_out = self.discriminator_loss(*args)

    if self.verbose:
      for loss_name in self.current_losses:
        loss_float = self.current_losses[loss_name]
        self.writer.add_scalar(f"{loss_name} loss", loss_float, self.step_id)
    self.writer.add_scalar("discriminator total loss", float(loss_val), self.step_id)

    loss_val.backward()
    self.discriminator_optimizer.step()

  def generator_step(self, data):
    """Performs a single step of generator training.

    Args:
      data: data points used for training.
    """
    self.generator_optimizer.zero_grad()
    data = to_device(data, self.device)
    args = self.run_generator(data)
    loss_val = self.generator_loss(*args)

    if self.verbose:
      if self.step_id % self.report_steps == 0:
        self.each_generate(*args)
      for loss_name in self.current_losses:
        loss_float = self.current_losses[loss_name]
        self.writer.add_scalar(f"{loss_name} loss", loss_float, self.step_id)
    self.writer.add_scalar("generator total loss", float(loss_val), self.step_id)

    loss_val.backward()
    self.generator_optimizer.step()

  def step(self, data):
    """Performs a single step of GAN training, comprised of
    one or more steps of discriminator and generator training.

    Args:
      data: data points used for training."""
    for _ in range(self.n_critic):
      self.discriminator_step(next(data))
    for _ in range(self.n_actor):
      self.generator_step(next(data))
    self.each_step()

  def checkpoint(self):
    """Performs a checkpoint of all generators and discriminators."""
    for name in self.generator_names:
      the_net = getattr(self, name)
      if isinstance(the_net, torch.nn.DataParallel):
        the_net = the_net.module
      netwrite(
        the_net,
        f"{self.checkpoint_path}-{name}-epoch-{self.epoch_id}-step-{self.step_id}.torch"
      )
    for name in self.discriminator_names:
      the_net = getattr(self, name)
      if isinstance(the_net, torch.nn.DataParallel):
        the_net = the_net.module
      netwrite(
        the_net,
        f"{self.checkpoint_path}-{name}-epoch-{self.epoch_id}-step-{self.step_id}.torch"
      )
    self.each_checkpoint()

  def train(self):
    """Trains a GAN until the maximum number of epochs is reached."""
    for epoch_id in range(self.max_epochs):
      self.epoch_id = epoch_id
      self.train_data = None
      self.train_data = DataLoader(
        self.data, batch_size=self.batch_size, num_workers=8,
        shuffle=True, drop_last=True
      )

      batches_per_step = self.n_actor + self.n_critic
      steps_per_episode = len(self.train_data) // batches_per_step

      data = iter(self.train_data)
      for _ in range(steps_per_episode):
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
  """Standard GAN training setup."""
  def __init__(self, generator, discriminator, data, **kwargs):
    """Standard setup of a generator and discriminator
    neural network playing a minimax game towards minimization
    of the Jensen-Shannon entropy.

    Args:
      generator (nn.Module): generator neural network.
      discriminator (nn.Module): discriminator neural network.
      data (Dataset): dataset providing real data.
      kwargs (dict): keyword arguments for generic GAN training procedures.
    """
    self.generator = ...
    self.discriminator = ...
    super(GANTraining, self).__init__(
      {"generator": generator},
      {"discriminator": discriminator},
      data, **kwargs
    )

  def sample(self):
    the_generator = self.generator
    if isinstance(the_generator, nn.DataParallel):
      the_generator = the_generator.module
    return to_device(the_generator.sample(self.batch_size), self.device)

  def generator_loss(self, data, generated):
    disc = self.discriminator(generated)
    loss_val = func.binary_cross_entropy_with_logits(
      disc, torch.ones(disc.size(0), 1).to(self.device)
    )

    return loss_val

  def discriminator_loss(self, generated, real,
                         generated_result, real_result):
    generated_loss = func.binary_cross_entropy_with_logits(
      generated_result, torch.zeros(generated_result.size(0), 1).to(self.device)
    )
    real_loss = func.binary_cross_entropy_with_logits(
      real_result, 0.9 * torch.ones(real_result.size(0), 1).to(self.device)
    )

    return generated_loss + real_loss

  def run_generator(self, data):
    sample = self.sample()
    generated = self.generator(sample)
    return data, generated

  def run_discriminator(self, data):
    with torch.no_grad():
      _, fake_batch = self.run_generator(data)
    fake_result = self.discriminator(fake_batch)
    real_result = self.discriminator(data)
    return data, fake_batch, fake_result, real_result

def _mix_on_path_aux(real, fake):
  sample = torch.rand(
    real.size(0),
    *[1 for _ in range(real.dim() - 1)]
  ).to(real.device)
  return (real * sample + fake * (1 - sample)).requires_grad_(True)

def _mix_on_path(real, fake):
  result = None
  if isinstance(real, (list, tuple)):
    result = [
      _mix_on_path(real_part, fake_part)
      for real_part, fake_part in zip(real, fake)
    ]
  elif isinstance(real, dict):
    result = {
      key: _mix_on_path(real[key], fake[key])
      for key in real
    }
  elif isinstance(real, torch.Tensor):
    if real.dtype in (torch.half, torch.float, torch.double):
      result = _mix_on_path_aux(real, fake)
    else:
      result = random.choice([real, fake])
  else:
    result = random.choice([real, fake])
  return result

def _gradient_norm(inputs, parameters):
  out = torch.ones(inputs.size()).to(inputs.device)
  gradients = torch.autograd.grad(
    inputs, parameters, create_graph=True, retain_graph=True,
    grad_outputs=out
  )
  grad_sum = 0.0
  for gradient in gradients:
    grad_sum += (gradient ** 2).view(gradient.size(0), -1).sum(dim=1)
  grad_sum = torch.sqrt(grad_sum)
  return grad_sum, out

class WGANTraining(GANTraining):
  """Wasserstein-GAN (Arjovsky et al. 2017) training setup
  with gradient penalty (Gulrajani et al. 2017) for more
  stable training."""
  def __init__(self, generator, discriminator, data, penalty=10, **kwargs):
    """Wasserstein-GAN training setup with gradient penalty,
    allowing for more stable training compared to standard GAN.

    Args:
      generator (nn.Module): generator neural network.
      discriminator (nn.Module): discriminator neural network.
      data (Dataset): dataset providing real data.
      penalty (float): coefficient of the gradient penalty.
      kwargs (dict): keyword arguments for generic GAN training procedures.
    """
    super(WGANTraining, self).__init__(generator, discriminator, data, **kwargs)
    self.penalty = penalty

  def mixing_key(self, data):
    return data

  def discriminator_loss(self, fake, real, generated_result, real_result):
    loss_val = generated_result.mean() - real_result.mean()
    mixed = _mix_on_path(real, fake)
    mixed_result = self.discriminator(mixed)
    grad_norm, out = _gradient_norm(mixed_result, self.mixing_key(mixed))
    gradient_penalty = ((grad_norm - 1) ** 2).mean()

    self.current_losses["discriminator"] = float(loss_val)
    self.current_losses["gradient-penalty"] = float(gradient_penalty)

    return loss_val + self.penalty * gradient_penalty, out

  def generator_loss(self, data, generated):
    return -self.discriminator(generated).mean()

class VKLGANTraining(WGANTraining):
  def discriminator_loss(self, fake, real, generated_result, real_result):
    real_mean = real_result.mean()
    generated_mean = generated_result.mean()
    loss_val = -generated_mean + real_mean
    real_weight = max(real_mean ** 2 - 1, 0.0)
    generated_weight = max(generated_mean ** 2 - 1, 0.0)
    penalty = real_weight + generated_weight

    mixed = _mix_on_path(real, fake)
    mixed_result = self.discriminator(mixed)
    grad_norm, out = _gradient_norm(mixed_result, self.mixing_key(mixed))
    cm = (1.0 / grad_norm).mean()

    self.current_losses["cm"] = float(cm)
    self.current_losses["discriminator"] = float(loss_val)
    self.current_losses["penalty"] = float(penalty)

    return loss_val + penalty, out

  def generator_loss(self, data, generated):
    return self.discriminator(generated).mean()

class RothGANTraining(GANTraining):
  def __init__(self, *args, gamma=0.1, **kwargs):
    super(RothGANTraining, self).__init__(*args, **kwargs)
    self.gamma = gamma

  def mixing_key(self, data):
    return data

  def regularization(self, fake, real, generated_result, real_result):
    fake_prob = torch.sigmoid(generated_result)
    real_prob = torch.sigmoid(real_result)
    real_norm, real_out = _gradient_norm(real_result, self.mixing_key(real))
    fake_norm, fake_out = _gradient_norm(generated_result, self.mixing_key(fake))

    real_penalty = real_norm ** 2 * (1 - real_prob) ** 2
    fake_penalty = fake_norm ** 2 * fake_prob ** 2

    penalty = 0.5 * self.gamma * (real_penalty + fake_penalty).mean()

    out = (real_out, fake_out)

    return penalty, out

  def discriminator_loss(self, fake, real, generated_result, real_result):
    loss_val = GANTraining.discriminator_loss(
      self, fake, real, generated_result, real_result
    )
    penalty, out = self.regularization(
      fake, real, generated_result, real_result
    )

    self.current_losses["discriminator"] = float(loss_val)
    self.current_losses["penalty"] = float(penalty)

    return loss_val + penalty, out

class GPGANTraining(GANTraining):
  """GAN training setup with zero-centered gradient penalty
  (Thanh-Tung et al. 2019) for more stable training of standard GAN."""
  def __init__(self, generator, discriminator, data, penalty=10, **kwargs):
    """GAN training setup with zero-centered gradient penalty,
    allowing for more stable training compared to standard GAN,
    without having to resort to using Wasserstein-1 distance.

    Args:
      generator (nn.Module): generator neural network.
      discriminator (nn.Module): discriminator neural network.
      data (Dataset): dataset providing real data.
      penalty (float): coefficient of the gradient penalty.
      kwargs (dict): keyword arguments for generic GAN training procedures.
    """
    super(GPGANTraining, self).__init__(generator, discriminator, data, **kwargs)
    self.penalty = penalty

  def mixing_key(self, data):
    return data

  def discriminator_loss(self, fake, real, generated_result, real_result):
    loss_val = torch.mean(real_result - generated_result)
    mixed = _mix_on_path(real, fake)
    mixed_result = self.discriminator(mixed)
    grad_norm, out = _gradient_norm(mixed_result, self.mixing_key(mixed))
    gradient_penalty = grad_norm ** 2

    self.current_losses["discriminator"] = float(loss_val)
    self.current_losses["gradient-penalty"] = float(gradient_penalty)

    return loss_val + self.penalty * gradient_penalty, out
