import random
from functools import partial
from torchsupport.flex.log.log_types import LogImage

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.data.namedtuple import namespace

def density_ratio_estimation(real, fake):
  real = func.binary_cross_entropy_with_logits(
    real, torch.ones_like(real), reduction="none"
  )
  fake = func.binary_cross_entropy_with_logits(
    fake, torch.zeros_like(fake), reduction="none"
  )
  return real + fake

def noise_contrastive_estimation(real, fake, real_base, fake_base, eps=1e-6):
  rlog = real - real_base
  flog = fake - fake_base
  rval = func.binary_cross_entropy_with_logits(
    rlog, torch.ones_like(rlog), reduction="none")
  fval = func.binary_cross_entropy_with_logits(
    flog, torch.zeros_like(flog), reduction="none")
  return rval + fval

def probability_surface_estimation(real, fake, real_base, fake_base, alpha=1/4):
  rfactor = (alpha * (real - real_base)).sigmoid() ** (1 / alpha)
  ffactor = (-alpha * (fake - fake_base)).sigmoid() ** (1 / alpha)
  return -(rfactor.detach() * real) + (ffactor.detach() * fake)

def tdre_mixing(data, sample, level):
  expand = [1] * (data.dim() - level.dim())
  level = level.view(*level.shape, *expand)
  return (1 - level ** 2).sqrt() * data + level * sample

def vp_mixing(data, sample, level):
  expand = [1] * (data.dim() - level.dim())
  level = level.view(*level.shape, *expand)
  return (1 - level).sqrt() * data + level.sqrt() * sample

def direct_mixing(data, sample, level):
  expand = [1] * (data.dim() - level.dim())
  level = level.view(*level.shape, *expand)
  return (1 - level) * data + level * sample

def ve_mixing(data, sample, level):
  expand = [1] * (data.dim() - level.dim())
  level = level.view(*level.shape, *expand)
  return data + level * sample

def random_dim_mixing(data, sample, level):
  mask = (torch.rand_like(data) < level).float()
  return mask * sample + (1 - mask) * data

# TODO: distribute over tuples?

def compose_mixing(data, base, mixers):
  real = []
  fake = []
  real_levels = []
  fake_levels = []
  for d, b, m in zip(data, base, mixers):
    r, f, rl, fl = m(d, b)
    real.append(r)
    fake.append(f)
    real_levels.append(rl)
    fake_levels.append(fl)
  return real, fake, real_levels, fake_levels

def independent_mixing(data, base, levels=None, mixing=vp_mixing, samples=1):
  levels = torch.cat((levels, torch.ones(1, device=levels.device)), dim=0)
  index = torch.randint(len(levels) - 1, (samples, data.size(0),), device=data.device)

  real_levels = levels[index]
  fake_levels = levels[index + 1]
  sample = base.sample(data.size(0) * samples)
  sample = sample.view(samples, *data.shape)

  real = mixing(data[None], sample, real_levels)
  fake = mixing(data[None], sample, fake_levels)

  real = real.view(-1, *real.shape[2:])
  fake = fake.view(-1, *fake.shape[2:])
  real_levels = real_levels.view(-1, *real_levels.shape[2:])
  fake_levels = fake_levels.view(-1, *fake_levels.shape[2:])

  return real, fake, real_levels, fake_levels

def coupled_mixing(data, base, levels=None, mixing=vp_mixing, subsample=None):
  sample = base.sample(data.size(0))
  expand = (data.dim() - 1) * [1]
  levels = levels.view(levels.size(0), *expand)
  mixed = torch.cat((
    mixing(data[None], sample[None], levels),
    sample[None]
  ), dim=0)
  real = mixed[:-1]
  fake = mixed[1:]
  if subsample is not None:
    index = torch.randperm(fake.size(0), device=fake.device)[:subsample]
    real = real[index]
    fake = fake[index]
    levels = levels[index]
  real = real.view(-1, *data.shape[1:])
  fake = fake.view(-1, *data.shape[1:])
  real_levels = levels.view(-1).repeat_interleave(data.size(0), dim=0)
  fake_levels = torch.cat((levels[1:], torch.ones(1, device=levels.device)), dim=0)
  fake_levels = fake_levels.view(-1).repeat_interleave(data.size(0), dim=0)
  return real, fake, real_levels, fake_levels

def run_density_ratio(energy, sample, data, args):
  real_data = data
  fake_data = sample
  real, fake = energy(real_data, args), energy(fake_data, args)
  loss = density_ratio_estimation(real, fake)
  return loss, namespace(
    real_data=real_data, fake_data=fake_data,
    real=real, fake=fake
  )

def density_ratio_step(energy, base, data, ctx=None):
  data, condition = data.sample(ctx.batch_size)
  sample = base.sample(ctx.batch_size)
  loss, args = run_density_ratio(energy, sample, data, condition)
  ctx.argmin(density_ratio_loss=loss)
  return args

def log_levels(name, data, levels, ctx=None):
  with torch.no_grad():
    for level in levels.unique():
      mask = (levels == level)
      mask_mean = data[mask].mean()
      ctx.log(**{f"{name} {level:.3}": float(mask_mean)})

def run_tdre(energy, base, data, args, mixing=None):
  real_data, fake_data, levels, _ = mixing(data, base)
  real, fake = energy(real_data, levels, args), energy(fake_data, levels, args)
  level_losses = density_ratio_estimation(real, fake)
  return level_losses.mean(), namespace(
    real_data=real_data, fake_data=fake_data,
    real=real, fake=fake, levels=levels, level_losses=level_losses
  )

def tdre_step(energy, base, data, mixing=None, ctx=None):
  data, condition = data.sample(ctx.batch_size)
  loss, args = run_tdre(energy, base, data, condition, mixing=mixing)
  ctx.argmin(density_ratio_loss=loss)
  log_levels(
    "level loss", args.level_losses, args.levels, ctx=ctx)
  return args

def run_tnce(energy, base, data, args, mixing=None,
             noise_contrastive=probability_surface_estimation):
  real_data, fake_data, real_levels, fake_levels = mixing(data, base)
  real = energy(real_data, real_levels, args)
  fake = energy(fake_data, real_levels, args)
  real_base = energy(real_data, fake_levels, args)
  fake_base = energy(fake_data, fake_levels, args)
  is_base = (fake_levels == 1.0)[:, None]
  base_real = base.log_prob(real_data, args)
  base_fake = base.log_prob(fake_data, args)
  real_base = (~is_base).float() * real_base + is_base.float() * base_real
  fake_base = (~is_base).float() * fake_base + is_base.float() * base_fake
  level_losses = noise_contrastive(real, fake, real_base, fake_base)
  return level_losses.mean(), namespace(
    real_data=real_data, fake_data=fake_data,
    real=real, fake=fake, level_losses=level_losses,
    real_levels=real_levels, fake_levels=fake_levels
  )

def tnce_step(energy, base, data, mixing=None,
              noise_contrastive=probability_surface_estimation,
              ctx=None):
  data, condition = data.sample(ctx.batch_size)
  loss, args = run_tnce(
    energy, base, data, condition,
    mixing=mixing, noise_contrastive=noise_contrastive
  )
  ctx.argmin(density_ratio_loss=loss)
  log_levels(
    "level loss", args.level_losses, args.real_levels, ctx=ctx)
  return args
