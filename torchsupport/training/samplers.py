import random

import numpy as np

import torch
from torch import nn
from torch.nn import functional as func
import torch.autograd as ag

from torchsupport.modules.gradient import hard_one_hot
from torchsupport.data.io import make_differentiable, detach

def clip_grad_by_norm(gradient, max_norm=0.01):
  norm = torch.norm(gradient)
  if norm > max_norm:
    gradient = gradient * (max_norm / norm)
  return gradient

class Langevin(nn.Module):
  target = None
  step = 0
  def __init__(self, rate=100.0, noise=0.005, steps=10, max_norm=0.01, clamp=(0, 1)):
    super(Langevin, self).__init__()
    self.rate = rate
    self.noise = noise
    self.steps = steps
    self.max_norm = max_norm
    self.clamp = clamp

  def step(self, score, data, *args, diffable=True):
    make_differentiable(data)
    make_differentiable(args)
    # data = ...
    if isinstance(data, (list, tuple)):
      data = [
        item + noise * torch.randn_like(item)
        for noise, item in zip(self.noise, data)
      ]
    else:
      data = data + self.noise * torch.randn_like(data)
    energy = score(data, *args)
    if isinstance(energy, (list, tuple)):
      energy, *_ = energy

    gradient = ag.grad(energy, data, torch.ones_like(energy), create_graph=diffable)
    if isinstance(data, (list, tuple)):
      data = list(data)
      for idx, (rate, clamp, gradval) in enumerate(zip(
        self.rate, self.clamp, gradient
      )):
        data[idx] = data[idx] - rate * gradval
        if clamp is not None:
          data[idx] = data[idx].clamp(*clamp)
    else:
      gradient = gradient[0]
      if self.max_norm:
        gradient = clip_grad_by_norm(gradient, self.max_norm)
      data = data - self.rate * gradient
      if self.clamp is not None:
        if isinstance(self.clamp, (list, tuple)):
          data = data.clamp(*self.clamp)
        else:
          data = self.clamp(data)
    return data

  def integrate(self, score, data, *args):
    for idx in range(self.steps):
      data = self.step(score, data, *args, diffable=False)
      data = detach(data)
    return data

class AugmentedLangevin(Langevin):
  def __init__(self, rate=100.0, noise=0.005, steps=10, max_norm=0.01, clamp=(0,1),
               transform_interval=50, transform=None):
    super().__init__(rate=rate, noise=noise, steps=steps, max_norm=max_norm, clamp=clamp)
    self.transform_interval = transform_interval
    self.transform = transform or (lambda x: x)

  def integrate(self, score, data, *args):
    for idx in range(self.steps):
      last_step = idx == self.steps - 1
      if idx % self.transform_interval == 0 and not last_step:
        with torch.no_grad():
          data = self.transform(data)
      data = self.step(score, data, *args, diffable=False)
      #data = detach(data)
    return data

class PackedLangevin(Langevin):
  def integrate(self, score, data, *args):
    for idx in range(self.steps):
      make_differentiable(data)
      make_differentiable(args)
      energy = score(data, *args)
      if isinstance(energy, (list, tuple)):
        energy, *_ = energy
      gradient = ag.grad(energy, data.tensor, torch.ones_like(energy))[0]
      if self.max_norm:
        gradient = clip_grad_by_norm(gradient, self.max_norm)
      data.tensor = data.tensor - self.rate * gradient + self.noise * torch.randn_like(data.tensor)
      if self.clamp is not None:
        data.tensor = data.tensor.clamp(*self.clamp)
      data.tensor = data.tensor % (2 * np.pi)
    return data

class AdaptiveLangevin(Langevin):
  def integrate(self, score, data, *args):
    done = False
    count = 0
    step_count = self.steps if self.step > 0 else 10 * self.steps
    while not done:
      make_differentiable(data)
      make_differentiable(args)
      energy = score(data + self.noise * torch.randn_like(data), *args)
      if isinstance(energy, (list, tuple)):
        energy, *_ = energy
      gradient = ag.grad(energy, data, torch.ones_like(energy))[0]
      if self.max_norm:
        gradient = clip_grad_by_norm(gradient, self.max_norm)
      data = data - self.rate * gradient
      if self.clamp is not None:
        data = data.clamp(*self.clamp)
      data = data.detach()
      done = count >= step_count
      if self.target is not None:
        done = done and bool((energy.mean(dim=0) <= self.target).all())
      count += 1
      if (count + 1) % 500 == 0:
        data.random_()
    self.step += 1
    return data

class TrueLangevin(Langevin):
  def __init__(self, noise=0.015, gradient_factor=1, take_noise=True, steps=100, clamp=(0, 1)):
    super().__init__(rate=None, noise=noise, steps=steps, max_norm=None, clamp=clamp)
    self.take_noise = take_noise
    self.gradient_factor = gradient_factor

  def integrate(self, score, data, *args):
    for idx in range(self.steps):
      make_differentiable(data)
      make_differentiable(args)
      energy = score(data, *args)
      if isinstance(energy, (list, tuple)):
        energy, *_ = energy
      gradient = self.gradient_factor * ag.grad(energy, data, torch.ones_like(energy))[0]
      noise = self.noise * torch.randn_like(data) if self.take_noise else 0.0
      data = data - self.noise ** 2 / 2 * gradient + noise
      if self.clamp is not None:
        data = data.clamp(*self.clamp)
    return data

class DiscreteLangevin(Langevin):
  def update(self, data):
    out = data.permute(0, 2, 1).contiguous().view(-1, data.size(1))
    dmax = out.argmax(dim=1)
    result = torch.zeros_like(out)
    result[torch.arange(0, out.size(0)), dmax] = 1
    return result.view(data.size(0), data.size(2), -1).permute(0, 2, 1).contiguous()

  def integrate(self, score, data, *args):
    for idx in range(self.steps):
      make_differentiable(data)
      make_differentiable(args)
      energy, *_ = score(data + self.noise * torch.randn_like(data), *args)
      gradient = ag.grad(energy, data, torch.ones(*energy.shape, device=data.device))[0]
      if self.max_norm:
        gradient = clip_grad_by_norm(gradient, self.max_norm)
      data = self.update(data - self.rate * gradient)
    return data

class PackedDiscreteLangevin(Langevin):
  def integrate(self, score, data, *args):
    data = data.clone()
    current_energy, *_ = score(data, *args)
    for idx in range(self.steps):
      make_differentiable(data)
      make_differentiable(args)

      energy = score(data, *args)
      if isinstance(energy, (list, tuple)):
        energy, *_ = energy

      gradient = ag.grad(energy, data.tensor, torch.ones_like(energy))[0]
      if self.max_norm:
        gradient = clip_grad_by_norm(gradient, self.max_norm)

      # attempt at gradient based local update of discrete variables:
      grad_prob = (-500 * gradient).softmax(dim=1)
      new_prob = self.noise + self.rate * grad_prob + (1 - self.noise - self.rate) * data.tensor
      new_val = hard_one_hot(new_prob.log())
      data.tensor = new_val

      data = data.detach()

    return data

class PackedDiscreteGPLangevin(Langevin):
  def __init__(self, scale=50, rate=0.5, noise=0.1, steps=10):
    super().__init__(rate=rate, noise=noise, steps=steps, max_norm=None, clamp=None)
    self.scale = scale

  def integrate(self, score, data, *args):
    data = data.clone()
    result = data.clone()
    current_energy = score(data, *args)
    for idx in range(self.steps):
      make_differentiable(data)
      make_differentiable(args)

      energy, deltas = score(data, *args, return_deltas=True)

      # attempt at gradient based local update of discrete variables:
      grad_prob = torch.zeros_like(deltas)
      grad_prob[torch.arange(deltas.size(0)), deltas.argmax(dim=1)] = 1
      if self.scale is not None:
        grad_prob = (self.scale * deltas).softmax(dim=1)
      new_prob = self.noise + self.rate * grad_prob + (1 - self.noise - self.rate) * data.tensor
      new_val = hard_one_hot(new_prob.log())
      data.tensor = new_val

      data = data.detach()

    return data

class PackedHardDiscreteLangevin(PackedDiscreteGPLangevin):
  def integrate(self, score, data, *args):
    data = data.clone()
    result = data.clone()
    current_energy = score(data, *args)
    for idx in range(self.steps):
      energy, deltas = score(data, *args, return_deltas=True)

      # attempt at gradient based local update of discrete variables:
      grad_prob = torch.zeros_like(deltas)
      grad_prob[torch.arange(deltas.size(0)), deltas.argmax(dim=1)] = 1
      if self.scale is not None:
        grad_prob = (self.scale * deltas).softmax(dim=1)
      access = torch.rand(deltas.size(0), dtype=torch.float, device=deltas.device)
      access = access < self.rate
      data.tensor[access] = hard_one_hot(grad_prob[access].log())

      data = data.detach()

    return data

class IndependentSampler(PackedDiscreteGPLangevin):
  def pick_positions(self, neighbours, counts):
    locations = torch.zeros(neighbours.size(0), dtype=torch.uint8)
    starts = []
    independents = []
    total = 0
    for count in counts:
      count = int(count)
      start = random.randrange(0, count)
      for idx in range(count):
        index = total + (start + idx) % count
        if locations[index]:
          continue
        locations[neighbours[index]] = 1
        independents.append(index)

      total += count

    return torch.tensor(sorted(independents))

  def perturb(self, data):
    positions = (torch.rand(data.tensor.size(0)) < self.noise).view(-1).nonzero().view(-1)
    values = torch.randint(0, 20, (positions.size(0),))
    data.tensor[positions] = 0
    data.tensor[positions, values] = 1
    return data

  def integrate(self, score, data, *args):
    data = data.clone()
    result = data.clone()
    current_energy = score(data, *args)
    access_cache = []
    for idx in range(self.steps):
      data = self.perturb(data)

      energy, deltas = score(data, *args, return_deltas=True)

      # attempt at gradient based local update of discrete variables:
      grad_prob = torch.zeros_like(deltas)
      grad_prob[torch.arange(deltas.size(0)), deltas.argmax(dim=1)] = 1
      if self.scale is not None:
        grad_prob = (self.scale * deltas).softmax(dim=1)

      access = self.pick_positions(args[-2].connections, args[-1].counts)
      data.tensor[access] = hard_one_hot(grad_prob[access].log())

      data = data.detach()

    return data

class MCMC():
  def __init__(self, temperature=0.01, constrain=True, steps=10):
    self.temperature = temperature
    self.constrain = constrain
    self.steps = steps

  def metropolis(self, current, proposal):
    log_alpha = - (proposal - current) / self.temperature
    alpha = log_alpha.exp().view(-1)
    uniform = torch.rand_like(alpha)
    accept = uniform < alpha
    accepted = accept.nonzero().view(-1)
    return accepted

  def mutate(self, score, data, *args):
    count = random.randint(1, 2)
    result = data.clone()
    for idx in range(count):
      position = torch.randint(0, result.size(2), (result.size(0),))
      change = torch.randint(0, result.size(1), (result.size(0),))
      result[torch.arange(0, result.size(0)), :, position] = 0
      result[torch.arange(0, result.size(0)), change, position] = 1
    return result

  def integrate(self, score, data, *args):
    result = data.clone()
    current_energy = score(data, *args)
    first_energy = current_energy.clone()
    for idx in range(self.steps):
      proposal = self.mutate(score, data, *args)
      energy = score(proposal, *args)
      accepted = self.metropolis(current_energy, energy)
      data[accepted] = proposal[accepted]
      current_energy[accepted] = energy[accepted]
    if self.constrain:
      accepted = (current_energy < first_energy).view(-1).nonzero().view(-1)
      result[accepted] = data[accepted]
    else:
      result = data
    return result

class PackedMCMC(MCMC):
  def mutate(self, score, data, *args):
    result = data.clone()
    count = 0
    for length in data.lengths:
      n_mutations = random.randrange(0, length)
      data_slice = slice(count, count + length)

      position = torch.randint(0, length, (n_mutations,))
      change = torch.randint(0, 20, (n_mutations,))

      subview = result.tensor[data_slice]
      subview[position, :] = 0
      subview[position, change] = 1

      count += length
    return result

  def metropolis(self, current, proposal):
    log_alpha = - (proposal - current) / self.temperature
    alpha = log_alpha.exp().view(-1)
    uniform = torch.rand_like(alpha)
    accept = uniform < alpha
    return accept

  def integrate(self, score, data, *args):
    with torch.no_grad():
      membership = args[-1]
      result = data.clone()
      current_energy = score(data, *args)
      first_energy = current_energy.clone()
      for idx in range(self.steps):
        proposal = self.mutate(score, data, *args)
        energy = score(proposal, *args)
        accepted = self.metropolis(current_energy, energy)
        current_energy[accepted] = energy[accepted]
        accepted = torch.repeat_interleave(accepted, membership.counts)
        data.tensor[accepted] = proposal.tensor[accepted]
      if self.constrain:
        accepted = (current_energy < first_energy).view(-1)
        accepted = torch.repeat_interleave(accepted, membership.counts)
        result.tensor[accepted] = data.tensor[accepted]
      else:
        result = data
      return result

class GeneticAlgorithmMCMC(MCMC):
  def crossover(self, score, data, *args):
    result = data.clone()
    reorder = torch.randint(0, data.size(0), (data.size(0),))
    choose = torch.randint(0, 2, (data.size(0), 1, data.size(2))).to(torch.float)
    result = choose * result + (1 - choose) * result[reorder]
    return result

  def mutate(self, score, data, *args):
    mutated = MCMC.mutate(self, score, data, *args)
    crossed_over = self.crossover(score, data, *args)
    choice = torch.randint(0, 2, (data.size(0), 1, 1)).to(torch.float)
    result = choice * mutated + (1 - choice) * crossed_over
    return result

class GradientProposalMCMC(MCMC):
  def mutate(self, score, data, *args):
    result = data.clone()

    make_differentiable(result)
    make_differentiable(args)
    energy = score(result, *args)
    gradient = ag.grad(energy, result, torch.ones(*energy.shape, device=result.device))[0]

    # position choice
    position_gradient = -gradient.sum(dim=1)
    position_distribution = torch.distributions.Categorical(logits=position_gradient)
    position_proposal = position_distribution.sample()

    # change choice
    change_gradient = -gradient[
      torch.arange(0, gradient.size(0)),
      :,
      position_proposal
    ]
    change_distribution = torch.distributions.Categorical(logits=change_gradient)
    change_proposal = change_distribution.sample()

    # mutate:
    result[torch.arange(0, result.size(0)), :, position_proposal] = 0
    result[torch.arange(0, result.size(0)), change_proposal, position_proposal] = 1

    return result.detach()

class AnnealedLangevin(nn.Module):
  def __init__(self, noises, steps=100, epsilon=2e-5):
    super(AnnealedLangevin, self).__init__()
    self.noises = noises
    self.steps = steps
    self.epsilon = epsilon

  def integrate(self, score, data, *args):
    for noise in self.noises:
      step_size = self.epsilon * (noise / self.noises[-1]) ** 2
      noise = torch.ones(data.size(0), *((data.dim() - 1) * [1])).to(data.device) * noise
      for step in range(self.steps):
        gradient = score(data, noise, *args)
        update = step_size * gradient + np.sqrt(2 * step_size) * torch.randn_like(data)
        data = data + update
    return data

class AnnealedPackedLangevin(nn.Module):
  def __init__(self, noises, scale=1, steps=10, epsilon=2e-5):
    super(AnnealedPackedLangevin, self).__init__()
    self.noises = noises
    self.steps = steps
    self.epsilon = epsilon
    self.scale = scale

  def integrate(self, score, data, *args):
    for idx, noise in enumerate(self.noises):
      step_size = self.epsilon * (noise / self.noises[-1]) ** 2
      noise = torch.ones(data.tensor.size(0), *((data.tensor.dim() - 1) * [1])).to(data.tensor.device) * noise
      step_count = self.steps[idx] if isinstance(self.steps, (list, tuple)) else self.steps
      for step in range(step_count):
        data.tensor = data.tensor + np.sqrt(2 * step_size) * torch.randn_like(data.tensor)
        gradient = score(data, noise, *args)
        update = self.scale * step_size * gradient
        data.tensor = (data.tensor + update)# % 6.3
    return data
