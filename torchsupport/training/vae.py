
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as func
from torch.utils.data import DataLoader
from torch.distributions import Normal, RelaxedOneHotCategorical

from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

from torchsupport.training.training import Training
from torchsupport.modules.losses.vae import \
  vae_loss, beta_vae_loss, joint_vae_loss, factor_vae_loss, tc_discriminator_loss
from torchsupport.data.io import netwrite

class AbstractVAETraining(Training):
  """Abstract base class for VAE training."""
  def __init__(self, networks, data,
               optimizer=torch.optim.Adam,
               optimizer_kwargs=None,
               max_epochs=50,
               batch_size=128,
               device="cpu",
               network_name="network",
               verbose=False):
    super(AbstractVAETraining, self).__init__()

    self.verbose = verbose
    self.checkpoint_path = network_name

    netlist = []
    self.network_names = []
    for network in networks:
      self.network_names.append(network)
      network_object = networks[network]
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

  def loss(self, *args):
    """Abstract method. Computes the training loss."""
    raise NotImplementedError("Abstract")

  def sample(self, *args, **kwargs):
    """Abstract method. Samples from the latent distribution."""
    raise NotImplementedError("Abstract")

  def run_networks(self, data):
    """Abstract method. Runs neural networks at each step."""
    raise NotImplementedError("Abstract")

  def step(self, data):
    self.optimizer.zero_grad()
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

    if self.verbose:
      for loss_name in self.current_losses:
        loss_float = self.current_losses[loss_float]
        self.writer.add_scalar(f"{loss_name} loss", loss_float, self.step_id)
    self.writer.add_scalar("total loss", float(loss_val), self.step_id)

    loss_val.backward()
    self.optimizer.step()
    self.each_step()

  def checkpoint(self):
    for name in self.network_names:
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

    netlist = [
      getattr(self, name)
      for name in self.network_names
    ]

    return netlist

class VAETraining(AbstractVAETraining):
  def __init__(self, encoder, decoder, data, **kwargs):
    self.encoder = ...
    self.decoder = ...
    super(VAETraining, self).__init__({
      "encoder": encoder,
      "decoder": decoder
    }, data, **kwargs)

  def loss(self, mean, logvar, reconstruction, target):
    loss_val, (ce, kld) = vae_loss(
      (mean, logvar), reconstruction, target,
      keep_components=True
    )
    self.current_losses["cross-entropy"] = float(ce)
    self.current_losses["kullback-leibler"] = float(kld)
    return loss_val

  def sample(self, mean, logvar):
    distribution = Normal(mean, torch.exp(0.5 * logvar))
    sample = distribution.rsample()
    return sample

  def run_networks(self, data):
    _, mean, logvar = self.encoder(data)
    sample = self.sample(mean, logvar)
    reconstruction = self.decoder(sample)
    return mean, logvar, reconstruction, data

class JointVAETraining(AbstractVAETraining):
  def __init__(self, encoder, decoder, data,
               n_classes=3, ctarget=50, dtarget=5,
               gamma=1000, temperature=0.67,
               **kwargs):
    self.encoder = ...
    self.decoder = ...
    super(JointVAETraining, self).__init__({
      "encoder": encoder,
      "decoder": decoder
    }, data, **kwargs)
    self.n_classes = n_classes
    self.temperature = temperature
    self.ctarget = ctarget
    self.dtarget = dtarget
    self.gamma = gamma

  def loss(self, normal_parameters, categorical_parameters,
           reconstruction, target):
    loss_val, (ce, n_n, n_c) = joint_vae_loss(
      normal_parameters,
      categorical_parameters,
      reconstruction, target,
      beta_normal=self.gamma,
      beta_categorical=self.gamma,
      c_normal=self.step_id * (self.ctarget) * 0.00001,
      c_categorical=min(
        self.step_id * (self.dtarget) * 0.00001,
        np.log(categorical_parameters.size(-1))
      ),
      keep_components=True
    )

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

  def run_networks(self, data):
    _, mean, logvar, probabilities = self.encoder(data)
    sample, category = self.sample(mean, logvar, probabilities)
    reconstruction = self.decoder(sample, category)
    return (mean, logvar), probabilities, reconstruction, data

class FactorVAETraining(JointVAETraining):
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

  def step(self, data):
    data = data.to(self.device)
    sample_data, shuffle_data = data[:data.size(0) // 2], data[data.size(0) // 2:]
    _, mean, logvar = self.encoder(sample_data)
    _, shuffle_mean, shuffle_logvar = self.encoder(shuffle_data)
    sample = self.sample(mean, logvar)
    shuffle_sample = self.sample(shuffle_mean, shuffle_logvar)
    reconstruction = self.decoder(sample)

    loss_val = factor_vae_loss(
      (mean, logvar), (self.decoder, sample),
      reconstruction, data, gamma=self.gamma
    )
    self.writer.add_scalar("total loss", float(loss_val), self.step_id)

    self.optimizer.zero_grad()
    loss_val.backward()
    self.optimizer.step()

    self.discriminator_optimizer.zero_grad()
    discriminator_loss = tc_discriminator_loss(
      self.discriminator,
      sample.detach(),
      shuffle_sample.detach()
    )
    discriminator_loss.backward()
    self.discriminator_optimizer.step()

    self.writer.add_scalar("total loss", float(loss_val), self.step_id)
    self.each_step()
