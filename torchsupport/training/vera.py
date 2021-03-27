import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal
import torch.autograd as ag

from torchsupport.data.io import to_device, make_differentiable, detach
from torchsupport.data.collate import DataLoader, default_collate
from torchsupport.training.energy import AbstractEnergyTraining

class VERAPosterior(nn.Module):
  def __init__(self):
    super().__init__()
    self.standard_deviation = nn.Parameter(
      torch.tensor(0.01).log()
    )

  def posterior(self, latent):
    return Normal(latent, self.standard_deviation.exp() + 0.01)

  def forward(self):
    return self.standard_deviation.exp() + 0.01

class VERAGenerator(nn.Module):
  def __init__(self, generator, latent_shape=None):
    super().__init__()
    self.generator = generator
    self.latent_shape = latent_shape or [64]
    self.log_conditional_variance = nn.Parameter(
      torch.tensor(0.01).log()
    )
    self.prior_mean = nn.Parameter(torch.zeros(self.latent_shape), requires_grad=False)
    self.prior_var = nn.Parameter(torch.ones(self.latent_shape), requires_grad=False)

  @property
  def prior(self):
    return Normal(self.prior_mean, self.prior_var)

  @property
  def conditional_variance(self):
    return self.log_conditional_variance.exp().clamp(0.01, 0.05)

  def sample(self, *args, sample_shape=None):
    sample_shape = sample_shape or []
    if not isinstance(sample_shape, (list, tuple, torch.Size)):
      sample_shape = [sample_shape]
    sample_shape = torch.Size(sample_shape)
    total = sample_shape.numel()

    latents = to_device(self.prior.sample((total,)), self.log_conditional_variance.device)
    result = self(latents, *args)
    result = result.view(*sample_shape, *result.shape[1:])
    latents = latents.view(*sample_shape, *latents.shape[1:])
    return latents, result

  def conditional(self, mean):
    return Normal(mean, self.conditional_variance)

  def forward(self, *args, **kwargs):
    return self.generator.forward(*args, **kwargs)

class VERATraining(AbstractEnergyTraining):
  def __init__(self, score, generator, *args,
               gradient_decay=0.1, hessian_decay=0.001, decay=1.0, k=20,
               entropy_weight=100,
               latent_shape=64,
               integrator=None,
               optimizer=torch.optim.Adam,
               generator_optimizer_kwargs=None,
               posterior_optimizer_kwargs=None,
               **kwargs):
    self.score = ...
    self.generator = ...
    self.posterior = ...
    super().__init__(
      {"score": score}, *args, optimizer=optimizer, **kwargs
    )

    generators = {"generator": VERAGenerator(generator, latent_shape=latent_shape)}
    self.generator_names, netlist = self.collect_netlist(generators)

    if generator_optimizer_kwargs is None:
      generator_optimizer_kwargs = {"lr": 5e-4}

    self.generator_optimizer = optimizer(
      netlist,
      **generator_optimizer_kwargs
    )

    posterior = {"posterior": VERAPosterior()}
    self.posterior_names, netlist = self.collect_netlist(posterior)

    if posterior_optimizer_kwargs is None:
      posterior_optimizer_kwargs = {"lr": 5e-4}

    self.posterior_optimizer = optimizer(
      netlist,
      **posterior_optimizer_kwargs
    )

    self.k = k
    self.entropy_weight = entropy_weight
    self.gradient_decay = gradient_decay
    self.hessian_decay = hessian_decay
    self.decay = decay
    self.integrator = integrator
    self.checkpoint_names.update(self.get_netlist(self.generator_names))

  def step(self, data):
    data = to_device(data, self.device)
    real, *args = self.data_key(data)
    latent, fake = self.posterior_step(real, args)
    self.energy_step(real, fake, args)
    self.generator_step(latent, fake, args)
    self.each_step()

  def run_posterior(self, data, args):
    latent, mean = self.generator.sample(*args, sample_shape=(self.batch_size,))
    post_latent = self.posterior.posterior(latent).rsample()
    post_mean = self.generator(post_latent, *args)
    prior = self.generator.prior.log_prob(post_latent).sum(dim=1)
    conditional = self.generator.conditional(post_mean).log_prob(mean).flatten(start_dim=1).sum(dim=1)
    posterior_entropy = self.posterior.posterior(latent.detach()).entropy().sum(dim=1)
    joint_log_probability = prior + conditional
    return latent, mean, joint_log_probability, posterior_entropy

  def posterior_loss(self, joint, entropy):
    result = -(joint.mean() + entropy.mean())
    self.current_losses["posterior"] = float(result)
    return result

  def posterior_step(self, real, args):
    self.posterior_optimizer.zero_grad()
    latent, mean, *args = self.run_posterior(real, args)
    loss = self.posterior_loss(*args)
    loss.backward(retain_graph=True)
    self.posterior_optimizer.step()
    return latent, mean

  def run_energy(self, real, fake, args):
    make_differentiable(real)
    real_result = self.score(real, *args)
    grad = ag.grad(
      real_result, real,
      grad_outputs=torch.ones_like(real_result),
      create_graph=True,
      retain_graph=True
    )[0]
    grad_norm = (grad.view(grad.size(0), -1) ** 2).sum(dim=1)

    fake_result = self.score(fake, *args)

    return real_result, fake_result, grad_norm

  def energy_loss(self, real, fake, grad):
    result = -real.mean() + fake.mean() + self.gradient_decay * grad.mean() + self.decay * ((real ** 2).mean() + (fake ** 2).mean())
    self.current_losses["energy"] = float(result)
    self.current_losses["fake"] = float(fake.mean())
    self.current_losses["real"] = float(real.mean())
    return result

  def energy_step(self, real, fake, args):
    self.optimizer.zero_grad()
    real_result, fake_result, grad_norm = self.run_energy(real, fake, args)
    loss = self.energy_loss(real_result, fake_result, grad_norm)
    loss.backward(retain_graph=True)
    self.optimizer.step()

  def run_generator(self, latent, fake, args):
    posterior = self.posterior.posterior(latent)
    new_latents = posterior.sample(sample_shape=(self.k,))
    new_latents = new_latents.view(-1, *latent.shape[1:])
    new_fake = self.generator(new_latents, *args)

    # compute importance weights
    log_conditional = self.generator.conditional(new_fake).log_prob(
      fake.repeat_interleave(self.k, dim=0)
    ).view(self.k, latent.size(0), -1).sum(dim=2)
    new_latents = new_latents.view(self.k, -1, *new_latents.shape[1:])
    log_prior = self.generator.prior.log_prob(new_latents).sum(dim=2)
    log_posterior = posterior.log_prob(new_latents).sum(dim=2)
    weight = log_prior + log_conditional - log_posterior
    weight = weight.softmax(dim=0)

    # compute score function estimator:
    grad = fake[None] - new_fake.view(self.k, *fake.shape)
    grad = grad / self.generator.conditional_variance ** 2
    grad = grad.flatten(start_dim=2)
    score = (weight[:, :, None] * grad).sum(dim=0).detach()

    # compute entropy gradient
    entropy_gradient = (score * fake.flatten(start_dim=1)).sum(dim=1).mean()
    fake_result = self.score(fake, *args)
    return fake_result, entropy_gradient

  def generator_loss(self, fake_result, entropy):
    result = -(fake_result.mean() + self.entropy_weight * entropy)
    self.current_losses["generator"] = float(result)
    return result

  def generator_step(self, latent, fake, args):
    self.generator_optimizer.zero_grad()
    fake_result, entropy = self.run_generator(latent, fake, args)
    loss = self.generator_loss(fake_result, entropy)
    self.log_statistics(float(loss))
    loss.backward()
    self.generator_optimizer.step()

  def sample(self):
    batch = next(iter(self.train_data))
    _, *args = self.data_key(batch)
    args = to_device(args, self.device)
    _, fake = self.generator.sample(*args, sample_shape=self.batch_size)
    improved = self.integrator.integrate(self.score, fake, *args).detach()
    return (fake, improved, *args)

  def each_step(self):
    super().each_step()
    if self.step_id % self.report_interval == 0 and self.step_id != 0:
      data, *args = detach(to_device(self.sample(), "cpu"))
      self.each_generate(data, *args)

  def each_generate(self, fake, improved, *args):
    fake = fake.sigmoid()
    improved = improved.sigmoid()
    self.writer.add_images("model", fake[:, None].repeat_interleave(3, dim=1), self.step_id)
    self.writer.add_images("energy", improved[:, None].repeat_interleave(3, dim=1), self.step_id)

  def data_key(self, data):
    result, *args = data
    return (result, *args)
