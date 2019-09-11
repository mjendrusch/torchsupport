from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
from torch.nn import functional as func
from torch.distributions import Normal, RelaxedOneHotCategorical

from tensorboardX import SummaryWriter

from torchsupport.training.state import (
  TrainingState, NetNameListState, NetState
)
from torchsupport.training.training import Training
import torchsupport.modules.losses.vae as vl
from torchsupport.data.io import netwrite, to_device
from torchsupport.data.collate import DataLoader

class AbstractVAETraining(Training):
  """Abstract base class for VAE training."""
  checkpoint_parameters = Training.checkpoint_parameters + [
    TrainingState(),
    NetNameListState("network_names"),
    NetState("optimizer")
  ]
  def __init__(self, networks, data, valid=None,
               optimizer=torch.optim.Adam,
               optimizer_kwargs=None,
               max_epochs=50,
               batch_size=128,
               device="cpu",
               network_name="network",
               verbose=False):
    """Generic training setup for variational autoencoders.

    Args:
      networks (list): networks used in the training step.
      data (Dataset): provider of training data.
      optimizer (Optimizer): optimizer class for gradient descent.
      optimizer_kwargs (dict): keyword arguments for the
        optimizer used in network training.
      max_epochs (int): maximum number of training epochs.
      batch_size (int): number of training samples per batch.
      device (string): device to use for training.
      network_name (string): identifier of the network architecture.
      verbose (bool): log all events and losses?
    """
    super(AbstractVAETraining, self).__init__()

    self.verbose = verbose
    self.checkpoint_path = network_name

    self.data = data
    self.valid = valid
    self.train_data = None
    self.valid_data = None
    self.max_epochs = max_epochs
    self.batch_size = batch_size
    self.device = device

    netlist = []
    self.network_names = []
    for network in networks:
      self.network_names.append(network)
      network_object = networks[network].to(self.device)
      setattr(self, network, network_object)
      netlist.extend(list(network_object.parameters()))

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

  def save_path(self):
    return f"{self.checkpoint_path}-save.torch"

  def divergence_loss(self, *args):
    """Abstract method. Computes the divergence loss."""
    raise NotImplementedError("Abstract")

  def reconstruction_loss(self, *args):
    """Abstract method. Computes the reconstruction loss."""
    raise NotImplementedError("Abstract")

  def loss(self, *args):
    """Abstract method. Computes the training loss."""
    raise NotImplementedError("Abstract")

  def sample(self, *args, **kwargs):
    """Abstract method. Samples from the latent distribution."""
    raise NotImplementedError("Abstract")

  def run_networks(self, data, *args):
    """Abstract method. Runs neural networks at each step."""
    raise NotImplementedError("Abstract")

  def preprocess(self, data):
    """Takes and partitions input data into VAE data and args."""
    return data

  def step(self, data):
    """Performs a single step of VAE training.

    Args:
      data: data points used for training."""
    self.optimizer.zero_grad()
    data = to_device(data, self.device)
    data, *netargs = self.preprocess(data)
    args = self.run_networks(data, *netargs)

    loss_val = self.loss(*args)

    if self.verbose:
      for loss_name in self.current_losses:
        loss_float = self.current_losses[loss_name]
        self.writer.add_scalar(f"{loss_name} loss", loss_float, self.step_id)
    self.writer.add_scalar("total loss", float(loss_val), self.step_id)

    loss_val.backward()
    self.optimizer.step()
    self.each_step()

    return float(loss_val)

  def valid_step(self, data):
    """Performs a single step of VAE validation.

    Args:
      data: data points used for validation."""
    with torch.no_grad():
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
      args = self.run_networks(data)

      loss_val = self.loss(*args)

    return float(loss_val)

  def checkpoint(self):
    """Performs a checkpoint for all encoders and decoders."""
    for name in self.network_names:
      the_net = getattr(self, name)
      if isinstance(the_net, torch.nn.DataParallel):
        the_net = the_net.module
      netwrite(
        the_net,
        f"{self.checkpoint_path}-{name}-epoch-{self.epoch_id}-step-{self.step_id}.torch"
      )
    self.each_checkpoint()

  def train(self):
    """Trains a VAE until the maximum number of epochs is reached."""
    for epoch_id in range(self.max_epochs):
      self.epoch_id = epoch_id
      self.train_data = None
      self.train_data = DataLoader(
        self.data, batch_size=self.batch_size, num_workers=8,
        shuffle=True
      )
      for data in self.train_data:
        self.step(data)
        self.step_id += 1
      self.checkpoint()
      if self.valid is not None:
        self.valid_data = DataLoader(
          self.valid, batch_size=self.batch_size, num_workers=8,
          shuffle=True
        )
        loss = 0.0
        for data in self.valid_data:
          loss += self.valid_step(data)
        loss /= len(self.valid)
        self.writer.add_scalar("valid loss", loss, self.step_id)

    netlist = [
      getattr(self, name)
      for name in self.network_names
    ]

    return netlist

class LaggingInference(ABC):
  """Mixin replacing the training method of a VAE training
  approach with the one detailed in "Lagging Inference Networks
  and Posterior Collapse in Variational Autoencoders"."""
  data = ...
  batch_size = ...
  encoder = ...
  decoder = ...
  device = ...
  max_epochs = ...
  valid = ...
  network_names = ...

  @abstractmethod
  def step(self, data):
    pass

  @abstractmethod
  def sample(self, *inputs):
    pass

  @abstractmethod
  def checkpoint(self):
    pass

  def aggressive_update(self, data):
    inner_data = DataLoader(
      self.data, batch_size=self.batch_size, num_workers=8,
      shuffle=True
    )
    for parameter in self.decoder:
      parameter.requires_grad = False
    last_ten = [None] * 10
    for idx, data_p in enumerate(inner_data):
      loss = self.step(data_p)
      last_ten[idx % 10] = loss
      if last_ten[-1] is not None and last_ten[-1] >= last_ten[0]:
        break
    for parameter in self.decoder:
      parameter.requires_grad = True
    for parameter in self.encoder:
      parameter.requires_grad = False
    self.step(data)
    for parameter in self.encoder:
      parameter.requires_grad = True

  def log_sum_exp(self, value, dim=None, keepdim=False):
    if dim is not None:
      m, _ = torch.max(value, dim=dim, keepdim=True)
      value0 = value - m
      if keepdim is False:
        m = m.squeeze(dim)
      return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
      m = torch.max(value)
      sum_exp = torch.sum(torch.exp(value - m))
    return m + torch.log(sum_exp)

  def compute_mi(self, x):
    """Approximate the mutual information between x and z
    I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))
    Returns: Float
    """
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

    with torch.no_grad():
      mu, logvar = self.encoder.forward(x)
      x_batch, nz = mu.size()
      neg_entropy = (-0.5 * nz * np.log(2 * np.pi) - 0.5 * (1 + logvar).sum(-1)).mean()

      z_samples = self.sample(mu, logvar)
      z_samples = z_samples[:, None, :]
      mu, logvar = mu[None], logvar[None]
      var = logvar.exp()

      dev = z_samples - mu
      log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
          0.5 * (nz * np.log(2 * np.pi) + logvar.sum(-1))
      log_qz = self.log_sum_exp(log_density, dim=1) - torch.log(x_batch)

      return (neg_entropy - log_qz.mean(-1)).item()

  def train(self):
    aggressive = True
    old_mi = 0
    new_mi = 0
    self.step_id = 0
    for epoch_id in range(self.max_epochs):
      self.epoch_id = epoch_id
      self.train_data = None
      self.train_data = DataLoader(
        self.data, batch_size=self.batch_size, num_workers=8,
        shuffle=True
      )
      for data in self.train_data:
        if aggressive:
          self.aggressive_update(data)
        else:
          self.step(data)
        self.step_id += 1
      valid_data = DataLoader(self.valid, batch_size=self.batch_size, shuffle=True)
      new_mi = self.compute_mi(next(iter(valid_data)))
      aggressive = new_mi > old_mi
      self.checkpoint()

    netlist = [
      getattr(self, name)
      for name in self.network_names
    ]

    return netlist

class AETraining(AbstractVAETraining):
  """Plain autoencoder training setup."""
  def __init__(self, autoencoder, data, **kwargs):
    self.autoencoder = ...
    super(AETraining, self).__init__({
      "autoencoder": autoencoder
    }, data, **kwargs)

  def loss(self, reconstruction, target):
    return vl.reconstruction_bce(reconstruction, target)

  def run_networks(self, data):
    reconstruction, *_ = self.autoencoder(data)
    return reconstruction, data

class VAETraining(AbstractVAETraining):
  """Standard VAE training setup."""
  def __init__(self, encoder, decoder, data, **kwargs):
    """Standard VAE training setup, training a pair of
    encoder and decoder to maximize the evidence lower
    bound.

    Args:
      encoder (nn.Module): encoder giving the variational posterior.
      decoder (nn.Module): decoder generating data from latent
        representations.
      data (Dataset): dataset providing training data.
      kwargs (dict): keyword arguments for generic VAE training.
    """
    self.encoder = ...
    self.decoder = ...
    super(VAETraining, self).__init__({
      "encoder": encoder,
      "decoder": decoder
    }, data, **kwargs)

  def reconstruction_loss(self, reconstruction, target):
    return vl.reconstruction_bce(reconstruction, target)

  def divergence_loss(self, mean, logvar):
    return vl.normal_kl_loss(mean, logvar)

  def loss(self, mean, logvar, reconstruction, target):
    ce = self.reconstruction_loss(reconstruction, target)
    kld = self.divergence_loss(mean, logvar)
    loss_val = ce + kld
    self.current_losses["cross-entropy"] = float(ce)
    self.current_losses["kullback-leibler"] = float(kld)
    return loss_val

  def sample(self, mean, logvar):
    distribution = Normal(mean, torch.exp(0.5 * logvar))
    sample = distribution.rsample()
    return sample

  def run_networks(self, data, *args):
    _, mean, logvar = self.encoder(data, *args)
    sample = self.sample(mean, logvar)
    reconstruction = self.decoder(sample, *args)
    return mean, logvar, reconstruction, data

class IntroVAETraining(VAETraining):
  def __init__(self, encoder, decoder, data,
               alpha=0.25, beta=0.5, m=120,
               optimizer=torch.optim.Adam,
               optimizer_kwargs=None, **kwargs):
    super(IntroVAETraining, self).__init__(
      encoder, decoder, data,
      optimizer=optimizer,
      optimizer_kwargs=optimizer_kwargs,
      **kwargs 
    )
    self.alpha = alpha
    self.beta = beta
    self.m = m
    self.optimizer = None
    self.generator_optimizer = optimizer(
      list(self.decoder.parameters()),
      **optimizer_kwargs
    )
    self.critic_optimizer = optimizer(
      list(self.encoder.parameters()),
      **optimizer_kwargs
    )

  def sample_prior(self, encoding):
    return torch.randn_like(encoding)

  def loss_adv(self, par, rt_par, p_rt_par):
    kld = self.divergence_loss(*par)
    rt_kld = self.divergence_loss(*rt_par)
    rt_kld = max(self.m - rt_kld, 0)
    p_rt_kld = self.divergence_loss(*rt_kld)
    p_rt_kld = max(self.m - p_rt_kld)
    return kld + self.alpha * (rt_kld + p_rt_kld)

  def loss_gen(self, rt_par, p_rt_par):
    rt_kld = self.divergence_loss(*rt_par)
    p_rt_kld = self.divergence_loss(*rt_kld)
    return self.alpha * (rt_kld + p_rt_kld)

  def critic_loss(self, par, rt_par, p_rt_par, reconstruction, target):
    ce = self.reconstruction_loss(reconstruction, target)
    reg = self.loss_adv(par, rt_par, p_rt_par)

    return ce + self.beta * reg

  def generator_loss(self, rt_par, p_rt_par, reconstruction, target):
    ce = self.reconstruction_loss(reconstruction, target)
    reg = self.loss_gen(rt_par, p_rt_par)

    return ce + self.beta * reg

  def run_critic(self, data, *args):
    _, *parameters = self.encoder(data, *args)
    sample = self.sample(*parameters)
    prior = self.sample_prior(sample)
    reconstruction = self.decoder(sample, *args)
    prior_reconstruction = self.decoder(prior, *args)
    _, *roundtrip_parameters = self.encoder(reconstruction.detach(), *args)
    _, *prior_roundtrip_parameters = self.encoder(prior_reconstruction.detach(), *args)

    return (
      (reconstruction, prior_reconstruction),
      parameters, roundtrip_parameters, prior_roundtrip_parameters, data, reconstruction
    )

  def run_generator(self, data, reconstruction, prior_reconstruction,  *args):
    _, *roundtrip_parameters = self.encoder(reconstruction *args)
    _, *prior_roundtrip_parameters = self.encoder(prior_reconstruction *args)
    return roundtrip_parameters, prior_roundtrip_parameters, data, reconstruction

  def step(self, data):
    data = to_device(data, self.device)
    data, *netargs = self.preprocess(data)
    
    self.critic_optimizer.zero_grad()
    pass_through, *critic_args = self.run_critic(data, *netargs)
    loss_val = self.critic_loss(*critic_args)
    loss_val.backward(retain_graph=True)
    self.writer.add_scalar("critic loss", float(loss_val), self.step_id)
    self.critic_optimizer.step()

    self.generator_optimizer.zero_grad()
    generator_args = self.run_generator(data, *pass_through, *netargs)
    loss_val = self.generator_loss(*generator_args)
    loss_val.backward()
    self.writer.add_scalar("generator loss", float(loss_val), self.step_id)
    self.generator_optimizer.step()

    if self.verbose:
      for loss_name in self.current_losses:
        loss_float = self.current_losses[loss_name]
        self.writer.add_scalar(f"{loss_name} loss", loss_float, self.step_id)

    self.each_step()

    return float(loss_val)

class JointVAETraining(VAETraining):
  """Joint training of continuous and categorical latent variables."""
  def __init__(self, encoder, decoder, data,
               n_classes=3, ctarget=50, dtarget=5,
               gamma=1000, temperature=0.67,
               **kwargs):
    """Joint training of continuous and categorical latent variables.

    Args:
      encoder (nn.Module): encoder neural network.
      decoder (nn.Module): decoder neural network.
      data (Dataset): dataset providing training data.
      c_target (float): target KL-divergence for continuous latent
        variables in nats.
      d_target (float): target KL-divergence for discrete latent
        variables in nats.
      gamma (float): scaling factor for KL-divergence constraints.
      temperature (float): temperature parameter of the concrete distribution.
      kwargs (dict): keyword arguments for generic VAE training.
    """
    super(JointVAETraining, self).__init__(
      encoder, decoder, data, **kwargs
    )
    self.n_classes = n_classes
    self.temperature = temperature
    self.ctarget = ctarget
    self.dtarget = dtarget
    self.gamma = gamma

  def divergence_loss(self, normal_parameters, categorical_parameters):
    normal = self.gamma * vl.normal_kl_norm_loss(
      *normal_parameters,
      c=self.step_id * (self.ctarget) * 0.00001
    )
    categorical = self.gamma * vl.gumbel_kl_norm_loss(
      categorical_parameters, c=min(
        self.step_id * (self.dtarget) * 0.00001,
        np.log(categorical_parameters.size(-1))
      )
    )
    return normal, categorical

  def loss(self, normal_parameters, categorical_parameters,
           reconstruction, target):
    ce = self.reconstruction_loss(reconstruction, target)
    n_n, n_c = self.divergence_loss(normal_parameters, categorical_parameters)
    loss_val = ce + n_n + n_c

    self.current_losses["cross-entropy"] = float(ce)
    self.current_losses["norm-normal"] = float(n_n)
    self.current_losses["norm-categorical"] = float(n_c)

    return loss_val

  def sample(self, mean, logvar, probabilities):
    normal = Normal(mean, torch.exp(0.5 * logvar))
    categorical = RelaxedOneHotCategorical(
      self.temperature,
      probabilities
    )
    return normal.rsample(), categorical.rsample()

  def run_networks(self, data, *args):
    _, mean, logvar, probabilities = self.encoder(data, *args)
    sample, category = self.sample(mean, logvar, probabilities)
    reconstruction = self.decoder(sample, category, *args)
    return (mean, logvar), probabilities, reconstruction, data

class FactorVAETraining(JointVAETraining):
  """Training setup for FactorVAE - VAE with disentangled latent space."""
  def __init__(self, encoder, decoder, discriminator, data,
               optimizer=torch.optim.Adam,
               loss=nn.CrossEntropyLoss(),
               max_epochs=50,
               batch_size=128,
               device="cpu",
               network_name="network",
               ctarget=50,
               dtarget=5,
               gamma=1000):
    """Training setup for FactorVAE - VAE with disentangled latent space.

    Args:
        encoder (nn.Module): encoder neural network.
        decoder (nn.Module): decoder neural network.
        discriminator (nn.Module): auxiliary discriminator
          for approximation of latent space total correlation.
        data (Dataset): dataset providing training data.
        c_target (float): target KL-divergence for continuous latent
          variables in nats.
        d_target (float): target KL-divergence for discrete latent
          variables in nats.
        gamma (float): scaling factor for KL-divergence constraints.
        temperature (float): temperature parameter of the concrete distribution.
        kwargs (dict): keyword arguments for generic VAE training.
    """
    super(FactorVAETraining, self).__init__(
      encoder, decoder, data,
      optimizer=optimizer,
      loss=loss,
      max_epochs=max_epochs,
      batch_size=batch_size,
      device=device,
      network_name=network_name,
      ctarget=ctarget,
      dtarget=dtarget,
      gamma=gamma
    )
    self.discriminator = discriminator.to(device)

    self.discriminator_optimizer = optimizer(
      self.discriminator.parameters(),
      lr=1e-4
    )

  def sample(self, mean, logvar):
    distribution = Normal(mean, torch.exp(0.5 * logvar))
    sample = distribution.rsample()
    return sample

  def divergence_loss(self, normal_parameters, decoder_parameters):
    tc_loss = vl.tc_encoder_loss(*decoder_parameters)
    div_loss = vl.normal_kl_loss(*normal_parameters)
    result = div_loss + self.gamma * tc_loss
    return result

  def step(self, data):
    data = data.to(self.device)
    sample_data, shuffle_data = data[:data.size(0) // 2], data[data.size(0) // 2:]
    _, mean, logvar = self.encoder(sample_data)
    _, shuffle_mean, shuffle_logvar = self.encoder(shuffle_data)
    sample = self.sample(mean, logvar)
    shuffle_sample = self.sample(shuffle_mean, shuffle_logvar)
    reconstruction = self.decoder(sample)

    ce = self.reconstruction_loss(reconstruction, data)
    tc = self.divergence_loss((mean, logvar), (self.decoder, sample))
    loss_val = ce + tc
    self.writer.add_scalar("total loss", float(loss_val), self.step_id)

    self.optimizer.zero_grad()
    loss_val.backward()
    self.optimizer.step()

    self.discriminator_optimizer.zero_grad()
    discriminator_loss = vl.tc_discriminator_loss(
      self.discriminator,
      sample.detach(),
      shuffle_sample.detach()
    )
    discriminator_loss.backward()
    self.discriminator_optimizer.step()

    self.writer.add_scalar("discriminator loss", float(discriminator_loss), self.step_id)
    self.each_step()

class ConditionalVAETraining(VAETraining):
  def __init__(self, encoder, decoder, prior, data, **kwargs):
    self.prior = ...
    self.encoder = ...
    self.decoder = ...
    AbstractVAETraining.__init__(self, {
      "encoder": encoder,
      "decoder": decoder,
      "prior": prior
    }, data, **kwargs)

  def preprocess(self, data):
    data, condition = data
    return data, condition

  def divergence_loss(self, parameters, prior_parameters):
    mu, lv = parameters
    mu_r, lv_r = prior_parameters
    result = vl.normal_kl_loss(mu, lv, mu_r, lv_r)
    return result

  def loss(self, parameters, prior_parameters, sample, reconstruction, target):
    ce = self.reconstruction_loss(reconstruction, target)
    kld = self.divergence_loss(parameters, prior_parameters)
    loss_val = ce + kld

    self.current_losses["cross-entropy"] = float(ce)
    self.current_losses["kullback-leibler"] = float(kld)

    return loss_val

  def sample(self, mean, logvar):
    distribution = Normal(mean, torch.exp(0.5 * logvar))
    sample = distribution.rsample()
    return sample

  def run_networks(self, target, condition):
    _, mean, logvar = self.encoder(target, condition)
    _, r_mean, r_logvar = self.prior(condition)
    sample = self.sample(mean, logvar)
    reconstruction = self.decoder(sample, condition)
    return (mean, logvar), (r_mean, r_logvar), sample, reconstruction, target

class GSSNConditionalVAETraining(ConditionalVAETraining):
  def __init__(self, encoder, decoder, prior, data,
               gssn=0.5, **kwargs):
    self.gssn = gssn
    super(GSSNConditionalVAETraining, self).__init__(
      encoder, decoder, prior, data, **kwargs
    )

  def loss(self, parameters, prior_parameters, sample, 
           reconstruction, r_reconstruction, target):
    loss_val = super().loss(
      parameters, prior_parameters, sample, reconstruction, target
    )
    gssn_val = func.binary_cross_entropy_with_logits(
      r_reconstruction, target, reduction='sum'
    ) / target.size(0)

    self.current_losses["gssn-cross-entropy"] = float(gssn_val)

    loss_val = (1 - self.gssn) * loss_val + self.gssn * gssn_val

    return loss_val

  def run_networks(self, target, condition):
    _, mean, logvar = self.encoder(target, condition)
    _, r_mean, r_logvar = self.prior(condition)
    sample = self.sample(mean, logvar)
    r_sample = self.sample(r_mean, r_logvar)
    reconstruction = self.decoder(sample, condition)
    r_reconstruction = self.decoder(r_sample, condition)
    return (
      (mean, logvar), (r_mean, r_logvar),
      sample, reconstruction, r_reconstruction, target
    )

class ConditionalVAESkipTraining(ConditionalVAETraining):
  def run_networks(self, target, condition):
    _, mean, logvar = self.encoder(target, condition)
    _, r_mean, r_logvar, skip = self.prior(condition)
    sample = self.sample(mean, logvar)
    reconstruction = self.decoder(sample, condition, skip)
    return (mean, logvar), (r_mean, r_logvar), sample, reconstruction, target

class GSSNConditionalVAESkipTraining(GSSNConditionalVAETraining):
  def run_networks(self, target, condition):
    _, mean, logvar = self.encoder(target, condition)
    _, r_mean, r_logvar, skip = self.prior(condition)
    sample = self.sample(mean, logvar)
    r_sample = self.sample(r_mean, r_logvar)
    reconstruction = self.decoder(sample, condition, skip)
    r_reconstruction = self.decoder(r_sample, condition, skip)
    return (
      (mean, logvar), (r_mean, r_logvar),
      sample, reconstruction, r_reconstruction, target
    )

class MDNPriorConditionalVAETraining(ConditionalVAETraining):
  def loss(self, parameters, prior_parameters, sample, reconstruction, target):
    vae_loss = vl.vae_loss(parameters, reconstruction, target)
    mdn_loss = vl.mdn_loss(prior_parameters, sample)

    self.current_losses['vae'] = float(vae_loss)
    self.current_losses['mdn'] = float(mdn_loss)

    return vae_loss + mdn_loss

class IndependentConditionalVAETraining(VAETraining):
  def __init__(self, encoder, decoder, data, **kwargs):
    super(IndependentConditionalVAETraining, self).__init__(
      encoder, decoder, data, **kwargs
    )

  def preprocess(self, data):
    data, condition = data
    return data, condition

class ConditionalRecurrentCanvasVAETraining(ConditionalVAETraining):
  def __init__(self, encoder, decoder, prior, data,
               iterations=10, has_state=None, **kwargs):
    super(ConditionalRecurrentCanvasVAETraining, self).__init__(
      encoder, decoder, prior, data, **kwargs
    )
    if has_state is None:
      has_state = {
        "encoder": True,
        "decoder": True,
        "prior": False
      }
    self.has_state = has_state
    self.iterations = iterations

  def loss(self, parameters, prior_parameters, approximation, target):
    loss_val = func.binary_cross_entropy_with_logits(
      approximation, target, reduction="sum"
    ) / target.size(0)
    for p, p_r in zip(parameters, prior_parameters):
      loss_val += vl.normal_kl_loss(*p, *p_r)
    return loss_val

  def run_networks(self, data):
    target, condition = data
    approximation = target
    if self.has_state["encoder"]:
      encoder_state = self.encoder.initial_state()
    if self.has_state["decoder"]:
      decoder_state = self.decoder.initial_state()
    if self.has_state["prior"]:
      prior_state = self.prior.initial_state()

    parameters = []
    prior_parameters = []

    for _ in range(self.iterations):
      if self.has_state["encoder"]:
        _, mean, logvar, encoder_state = self.encoder(approximation, condition, encoder_state)
      else:
        _, mean, logvar = self.encoder(approximation, condition)

      if self.has_state["prior"]:
        _, r_mean, r_logvar, prior_state = self.prior(approximation, condition, prior_state)
      else:
        _, r_mean, r_logvar = self.prior(approximation, condition)

      parameters.append((mean, logvar))
      prior_parameters.append((r_mean, r_logvar))

      sample = self.sample(mean, logvar)

      if self.has_state["decoder"]:
        reconstruction, decoder_state = self.decoder(sample, condition, decoder_state)
      else:
        reconstruction = self.decoder(approximation, condition)

      approximation = reconstruction + approximation

    return parameters, prior_parameters, approximation, target

class IndependentConditionalRecurrentCanvasVAETraining(IndependentConditionalVAETraining):
  def __init__(self, encoder, decoder, data,
               iterations=10, has_state=None, **kwargs):
    super(IndependentConditionalRecurrentCanvasVAETraining, self).__init__(
      encoder, decoder, data, **kwargs
    )
    if has_state is None:
      has_state = {
        "encoder": True,
        "decoder": True
      }
    self.has_state = has_state
    self.iterations = iterations

  def loss(self, parameters, approximation, target):
    loss_val = func.binary_cross_entropy_with_logits(
      approximation, target, reduction="sum"
    ) / target.size(0)
    for mean, logvar in parameters:
      loss_val += vl.normal_kl_loss(mean, logvar)
    return loss_val

  def run_networks(self, data):
    target, condition = data
    approximation = target
    if self.has_state["encoder"]:
      encoder_state = self.encoder.initial_state()
    if self.has_state["decoder"]:
      decoder_state = self.decoder.initial_state()

    parameters = []

    for _ in range(self.iterations):
      if self.has_state["encoder"]:
        _, mean, logvar, encoder_state = self.encoder(approximation, condition, encoder_state)
      else:
        _, mean, logvar = self.encoder(approximation, condition)
      parameters.append((mean, logvar))

      sample = self.sample(mean, logvar)

      if self.has_state["decoder"]:
        reconstruction, decoder_state = self.decoder(sample, condition, decoder_state)
      else:
        reconstruction = self.decoder(approximation, condition)

      approximation = reconstruction + approximation

    return parameters, approximation, target
