
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
    loss_val, (ce, kld) = vl.vae_loss(
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
    loss_val, (ce, n_n, n_c) = vl.joint_vae_loss(
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

    loss_val = vl.factor_vae_loss(
      (mean, logvar), (self.decoder, sample),
      reconstruction, data, gamma=self.gamma
    )
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

    self.writer.add_scalar("total loss", float(loss_val), self.step_id)
    self.each_step()

class ConditionalVAETraining(AbstractVAETraining):
  def __init__(self, encoder, decoder, prior, data, **kwargs):
    self.prior = ...
    self.encoder = ...
    self.decoder = ...
    super(ConditionalVAETraining, self).__init__({
      "encoder": encoder,
      "decoder": decoder,
      "prior": prior
    }, data, **kwargs)

  def loss(self, parameters, prior_parameters, sample, reconstruction, target):
    loss_val, (ce, kld) = vl.conditional_vae_loss(
      parameters, prior_parameters, reconstruction, target,
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
    target, condition = data
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

  def run_networks(self, data):
    target, condition = data
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
  def run_networks(self, data):
    target, condition = data
    _, mean, logvar = self.encoder(target, condition)
    _, r_mean, r_logvar, skip = self.prior(condition)
    sample = self.sample(mean, logvar)
    reconstruction = self.decoder(sample, condition, skip)
    return (mean, logvar), (r_mean, r_logvar), sample, reconstruction, target

class GSSNConditionalVAESkipTraining(GSSNConditionalVAETraining):
  def run_networks(self, data):
    target, condition = data
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
    super(IndependentConditionalVAETraining, self).__init__({
      "encoder": encoder,
      "decoder": decoder
    }, data, **kwargs)

  def run_networks(self, data):
    target, condition = data
    _, mean, logvar = self.encoder(target, condition)
    sample = self.sample(mean, logvar)
    reconstruction = self.decoder(sample, condition)
    return (mean, logvar), reconstruction, target

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
