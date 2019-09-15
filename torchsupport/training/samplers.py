import random

import numpy as np

import torch
from torch import nn
from torch.nn import functional as func
import torch.autograd as ag

from torchsupport.data.io import make_differentiable

def clip_grad_by_norm(gradient, max_norm=0.01):
  norm = torch.norm(gradient)
  if norm > max_norm:
    gradient = gradient * (max_norm / norm)
  return gradient

class Langevin(nn.Module):
  def __init__(self, rate=100.0, noise=0.005, steps=10, max_norm=0.01, clamp=(0, 1)):
    super(Langevin, self).__init__()
    self.rate = rate
    self.noise = noise
    self.steps = steps
    self.max_norm = max_norm
    self.clamp = clamp

  def integrate(self, score, data, *args):
    for idx in range(self.steps):
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
    print(gradient.view(energy.size(0), -1).mean(dim=1)[:5])
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
      noise = torch.ones(data.size(0), data.size(1), 1, 1, 1).to(data.device) * noise
      for step in range(self.steps):
        gradient = score(data, noise, *args)
        update = step_size * gradient + np.sqrt(2 * step_size) * torch.randn_like(data)
        data = data + update
    return (data - data.min()) / (data.max() - data.min())
