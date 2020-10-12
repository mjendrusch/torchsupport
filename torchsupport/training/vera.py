import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal
import torch.autograd as ag

from torchsupport.data.io import to_device, make_differentiable
from torchsupport.data.collate import DataLoader, default_collate
from torchsupport.training.energy import AbstractEnergyTraining

class VERAPosterior(nn.Module):
  def __init__(self, shape):
    super().__init__()
    self.standard_deviation = nn.Parameter(
      (torch.ones(*shape) * 0.1).log()
    )

  def forward(self):
    return self.standard_deviation.exp()

class VERATraining(AbstractEnergyTraining):
  def __init__(self, score, generator, *args,
               gradient_decay=0.1, hessian_decay=0.001, k=20,
               entropy_weight=100,
               integrator=None,
               optimizer=torch.optim.Adam,
               generator_optimizer_kwargs=None,
               standard_deviation_optimizer_kwargs=None,
               **kwargs):
    self.score = ...
    self.generator = ...
    self.standard_deviation = ...
    super().__init__(
      {"score": score}, *args, optimizer=optimizer, **kwargs
    )

    generators = {"generator": generator}
    self.generator_names, netlist = self.collect_netlist(generators)

    if generator_optimizer_kwargs is None:
      generator_optimizer_kwargs = {"lr": 5e-4}

    self.generator_optimizer = optimizer(
      netlist,
      **generator_optimizer_kwargs
    )

    posterior = {"standard_deviation": VERAPosterior(self.score.sample(1).shape)}
    self.posterior_names, netlist = self.collect_netlist(posterior)

    if standard_deviation_optimizer_kwargs is None:
      standard_deviation_optimizer_kwargs = {"lr": 5e-4}

    self.standard_deviation_optimizer = optimizer(
      netlist,
      **standard_deviation_optimizer_kwargs
    )

    self.k = k
    self.entropy_weight = entropy_weight
    self.gradient_decay = gradient_decay
    self.hessian_decay = hessian_decay
    self.integrator = integrator
    self.checkpoint_names.update(self.get_netlist(self.generator_names))

  def step(self, data):
    data = to_device(data, self.device)
    real, *args = self.data_key(data)
    latent, fake = self.posterior_step(real, args)
    fake_result = self.energy_step(real, fake, args)
    self.generator_step(latent, fake, args, fake_result)
    super().step(data)

  def run_posterior(self, latent, data, args):
    latent_distribution = Normal(0.0, 1.0)
    mean = self.generator(latent, args)
    conditional_distribution = Normal(mean, self.generator.standard_deviation())
    posterior_distribution = Normal(latent.detach(), self.standard_deviation())
    joint_log_probability = latent_distribution.log_prob(latent).sum(dim=1) \
      + conditional_distribution.log_prob(data).flatten(start_dim=1).sum(dim=1)
    posterior_entropy = posterior_distribution.entropy().sum(dim=1)
    return mean, joint_log_probability, posterior_entropy

  def posterior_loss(self, mean, joint, entropy):
    result = -(joint.mean() + entropy.mean())
    self.current_losses["posterior"] = float(result)
    return result

  def posterior_step(self, real, args):
    self.standard_deviation_optimizer.zero_grad()
    latent = self.generator.sample(self.batch_size)
    mean, *args = self.run_posterior(latent, real, args)
    loss = self.posterior_loss(mean, *args)
    loss.backward(retain_graph=True)
    self.standard_deviation_optimizer.step()
    return latent, mean

  def run_energy(self, real, fake, args):
    make_differentiable(real)
    real_result = self.score(real, *args)
    grad = ag.grad(
      real_result, real,
      grad_outputs=torch.ones_like(real),
      create_graph=True,
      retain_graph=True
    )[0]
    grad_norm = (grad.view(grad.size(0), -1) ** 2).sum(dim=1)

    fake_result = self.score(fake, *args)

    return real_result, fake_result, grad_norm

  def energy_loss(self, real, fake, grad):
    result = real.mean() - fake.mean() + self.gradient_decay * grad.mean()
    self.current_losses["energy"] = float(result)
    return result

  def energy_step(self, real, fake, args):
    self.optimizer.zero_grad()
    real_result, fake_result, grad_norm = self.run_energy(real, fake, args)
    loss = self.energy_loss(real_result, fake_result, grad_norm)
    loss.backward(retain_graph=True)
    self.optimizer.step()
    return fake_result

  def run_generator(self, latent, fake, args):
    posterior = Normal(
      latent[None].repeat_interleave(self.k, dim=0),
      self.standard_deviation()
    )
    latent_distribution = Normal(0.0, 1.0)
    new_latents = posterior.sample()
    log_posterior = posterior.log_prob(new_latents).sum(dim=2)
    new_latents = new_latents.view(-1, latent.shape[1:])
    new_fake = self.generator(new_latents, args)

    distribution = Normal(new_fake, self.generator.standard_deviation())
    log_conditional = distribution.log_prob(
      fake[None].repeat_interleave(self.k, dim=0)
    ).view(self.k, latent.size(0), -1).sum(dim=2)
    log_prior = latent_distribution.log_prob(new_latents).sum(dim=2)
    weight = log_prior + log_conditional - log_posterior
    weight = weight.softmax(dim=0)
    value = fake[None] - new_fake.view(self.k, *fake.shape)
    value = value / self.generator.standard_deviation() ** 2
    value = value.flatten(start_dim=2)
    score = (weight[:, :, None] * value).sum(dim=0).detach()
    entropy_gradient = (score * fake.flatten(start_dim=1)).sum(dim=1).mean()
    return entropy_gradient

  def generator_loss(self, fake_result, entropy):
    result = fake_result.mean() - self.entropy_weight * entropy
    self.current_losses["generator"] = float(result)
    return result

  def generator_step(self, latent, fake, args, fake_result):
    self.generator_optimizer.zero_grad()
    entropy = self.run_generator(latent, fake, args)
    loss = self.generator_loss(fake_result, entropy)
    loss.backward()
    self.generator_optimizer.step()

  def each_step(self):
    super().each_step()
    # TODO
    # if self.step_id % self.report_interval == 0 and self.step_id != 0:
    #   data, *args = self.sample()
    #   self.each_generate(data, *args)

  def data_key(self, data):
    result, *args = data
    return (result, *args)
