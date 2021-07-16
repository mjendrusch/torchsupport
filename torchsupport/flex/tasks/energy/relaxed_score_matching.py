from functools import partial
from torchsupport.flex.tasks.energy.score_matching import linear_noise

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Distribution, Normal

from torchsupport.data.io import make_differentiable
from torchsupport.data.namedtuple import namespace
from torchsupport.data.collate import default_collate

class ProductNormal(Distribution):
  def __init__(self, mean, var):
    self._mean = mean
    self._var = var

  def sample(self, sample_shape=torch.Size()):
    return Normal(self._mean, self._var).sample(sample_shape)

  def log_prob(self, x):
    dist = Normal(self._mean, self._var)
    log_p = dist.log_prob(x)
    log_p = log_p.view(log_p.size(0), -1)
    return log_p.sum(dim=1, keepdim=True)

class TruncatedNormal(Distribution):
  def __init__(self, mean, var):
    self._mean = mean
    self._var = var 

  def sample(self, sample_shape=torch.Size()):
    result = Normal(torch.zeros_like(self._mean), self._var).sample(sample_shape)
    result = result.view(result.size(0), -1)
    norm = result.norm(dim=1, keepdim=True)
    result = result / (norm + 1e-3)
    result = result.view(*self._mean.shape)
    return result * self._var + self._mean

  def log_prob(self, x):
    dist = Normal(self._mean, self._var)
    log_p = dist.log_prob(x)
    log_p = log_p.view(log_p.size(0), -1)
    return log_p.sum(dim=1, keepdim=True)

class NormalNoise(nn.Module):
  def __init__(self, sigma=1e-3):
    super().__init__()
    self.sigma = sigma

  def forward(self, data, levels):
    sigma = self.sigma
    if not isinstance(sigma, float):
      expand = (data.dim() - 1) * [1]
      sigma = sigma(levels)
      sigma = sigma.view(sigma.size(0), *expand)
    return ProductNormal(data, sigma)

class VPNormalNoise(NormalNoise):
  def forward(self, data, levels):
    sigma = self.sigma
    if not isinstance(sigma, float):
      expand = (data.dim() - 1) * [1]
      sigma = sigma(levels)
      sigma = sigma.view(sigma.size(0), *expand)
    return ProductNormal((1 - sigma ** 2).sqrt() * data, sigma)

class LangevinNoise(nn.Module):
  def __init__(self, energy, sigma=1e-3, scale=1.0):
    super().__init__()
    self.sigma = sigma
    self.energy = energy
    self.scale = scale

  def mean(self, data, levels, sigma):
    make_differentiable(data)
    E = self.energy(data, levels)
    score = torch.autograd.grad(E, data, grad_outputs=torch.ones_like(E), retain_graph=True)[0]
    return (data + self.scale * sigma ** 2 * score / 2).detach()

  def forward(self, data, levels):
    sigma = self.sigma
    if not isinstance(sigma, float):
      expand = (data.dim() - 1) * [1]
      sigma = sigma(levels)
      sigma = sigma.view(sigma.size(0), *expand)
    mean = self.mean(data, levels, sigma)
    return ProductNormal(mean, self.scale * sigma)

class RecoveryNoise(LangevinNoise):
  def mean(self, data, levels, sigma):
    noised = (data + sigma * torch.randn_like(data)).detach()
    data = noised.clone()
    make_differentiable(data)
    E = self.energy(data, levels)
    E = E - ((data - noised) ** 2).view(data.size(0), -1).sum(dim=1, keepdim=True) / (2 * sigma ** 2)
    score = torch.autograd.grad(E, data, grad_outputs=torch.ones_like(E), retain_graph=True)[0]
    return (data + self.scale ** 2 * sigma ** 2 * score).detach()

class TruncatedNormalNoise(nn.Module):
  def __init__(self, sigma=1e-3):
    super().__init__()
    self.sigma = sigma

  def forward(self, data, levels):
    sigma = self.sigma
    if not isinstance(sigma, float):
      expand = (data.dim() - 1) * [1]
      sigma = sigma(levels)
      sigma = sigma.view(sigma.size(0), *expand)
    return TruncatedNormal(data, sigma)

class ReplacementDistribution(Distribution):
  def __init__(self, data, rate):
    self.data = data
    self.rate = rate

  def get_params(self, x):
    uniform = torch.ones_like(x) / x.size(1)
    log_probs = (self.rate * uniform + (1 - self.rate) * x).log()
    return log_probs

  def make_onehot(self):
    noise_index = torch.rand_like(self.data)
    noise_index = noise_index.transpose(1, 0).reshape(self.data.size(1), -1)
    noise = torch.zeros_like(noise_index)
    noise_index = noise_index.argmax(dim=0)
    ind = torch.arange(noise_index.size(1), dtype=torch.long, device=noise.device)
    noise[noise_index, ind] = 1
    noise = noise.reshape(self.data.size(1), self.data.size(0), *self.data.shape[2:])
    noise = noise.transpose(0, 1)
    return noise.contiguous()

  def sample(self, sample_shape=torch.Size()):
    random_onehot = self.make_onehot()
    mask = torch.rand(self.data.size(0), 1, *self.data.shape[2:]) < self.rate
    mask = mask.float()
    return random_onehot * mask + self.data * (mask - 1)

  def log_prob(self, x):
    log_probs = self.get_params(self.data)
    result = (x * log_probs).view(x.size(0), -1)
    return result.sum(dim=1, keepdim=True)

class ReplacementNoise(nn.Module):
  def __init__(self, rate=0.1):
    super().__init__()
    self.rate = rate

  def forward(self, data, levels):
    rate = self.rate
    if not isinstance(rate, float):
      expand = (data.dim() - 1) * [1]
      rate = rate(levels)
      rate = rate.view(rate.size(0), *expand)
    return ReplacementDistribution(data, self.rate)

def run_relaxed_score(energy, data, args, levels,
                      level_weight=1.0,
                      level_distribution=None,
                      noise_distribution=None):
  level_dist = level_distribution(data, levels)
  level_data = level_dist.sample()
  noise_dist = noise_distribution(level_data, levels)
  noise_data = noise_dist.sample()
  score = energy(noise_data, levels, args) - energy(level_data, levels, args)
  gt_score = level_dist.log_prob(noise_data) - level_dist.log_prob(level_data)
  if not isinstance(level_weight, float):
    level_weight = level_weight(levels)[:, None]
  loss = (level_weight * (score.exp() - gt_score.exp()) ** 2).mean()
  return loss, namespace(
    data=data, level_data=level_data, noise_data=noise_data,
    score=score, gt_score=gt_score, level_weight=level_weight
  )

def run_cnce(energy, data, args, levels,
             level_distribution=None,
             noise_distribution=None):
  level_dist = level_distribution(data, levels)
  level_data = level_dist.sample()
  noise_dist = noise_distribution(level_data, levels)
  noise_data = noise_dist.sample()
  data_dist = noise_distribution(noise_data, levels)
  weight = noise_dist.log_prob(noise_data) - data_dist.log_prob(level_data)
  score = -energy(noise_data, levels, args) + energy(level_data, levels, args)
  loss = -func.logsigmoid(weight + score).mean()
  return loss, namespace(
    data=data, level_data=level_data, noise_data=noise_data, score=score
  )

def run_recovery_likelihood(energy, data, args, levels,
                            level_weight=1.0,
                            level_distribution=None,
                            noise_distribution=None):
  level_dist = level_distribution(data, levels)
  level_data = level_dist.sample()
  noise_dist = noise_distribution(level_data, levels)
  noise_data = noise_dist.sample()
  loss = energy(noise_data, levels, args) - energy(level_data, levels, args)
  if not isinstance(level_weight, float):
    level_weight = level_weight(levels)[:, None]
  loss = (level_weight * loss).mean()
  return loss, namespace(
    data=data, level_data=level_data, noise_data=noise_data
  )

def relaxed_score_matching_step(energy, data,
                                level_weight=1.0,
                                level_distribution=NormalNoise(linear_noise()),
                                noise_distribution=NormalNoise(1e-3),
                                loss_scale=1e-3,
                                ctx=None):
  data, condition = data.sample(ctx.batch_size)
  levels = torch.rand(ctx.batch_size, device=ctx.device)
  loss, args = run_relaxed_score(
    energy, data, condition, levels,
    level_weight=level_weight,
    level_distribution=level_distribution,
    noise_distribution=noise_distribution
  )
  ctx.argmin(score_loss=loss_scale * loss)
  ctx.log(gt_score=float(args.gt_score.mean()))
  ctx.log(scaled_gt_score=float((args.gt_score ** 2 * args.level_weight).mean()))
  ctx.log(score=float(args.score.mean()))
  ctx.log(unscaled_loss=float(((args.gt_score - args.score) ** 2).mean()))
  return args

def cnce_step(energy, data,
              level_distribution=NormalNoise(linear_noise()),
              noise_distribution=NormalNoise(1e-3),
              ctx=None):
  data, condition = data.sample(ctx.batch_size)
  levels = torch.rand(ctx.batch_size, device=ctx.device)
  loss, args = run_cnce(
    energy, data, condition, levels,
    level_distribution=level_distribution,
    noise_distribution=noise_distribution
  )
  ctx.argmin(cnce_loss=loss)
  return args

def recovery_likelihood_step(energy, data,
                             level_weight=1.0,
                             loss_scale=1.0,
                             level_distribution=NormalNoise(linear_noise()),
                             noise_distribution=NormalNoise(1e-3),
                             ctx=None):
  data, condition = data.sample(ctx.batch_size)
  levels = torch.rand(ctx.batch_size, device=ctx.device)
  loss, args = run_recovery_likelihood(
    energy, data, condition, levels,
    level_weight=level_weight,
    level_distribution=level_distribution,
    noise_distribution=noise_distribution
  )
  ctx.argmin(energy_loss=loss_scale * loss)
  return args
