import random
from functools import partial
from torchsupport.flex.log.log_types import LogImage

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.data.namedtuple import namespace

def run_diffusion_recovery_likelihood(energy, base, data, args,
                                      integrator=None, mixing=None,
                                      conditional=None):
  real_data, condition, levels, _ = mixing(data, base)
  conditional_energy = conditional(energy, condition)
  fake_data = integrator.integrate(conditional_energy, condition, args)
  real, fake = energy(real_data, levels, args), energy(fake_data, levels, args)
  loss = real.mean() - fake.mean()
  return loss, namespace(
    real_data=real_data, fake_data=fake_data, condition=condition,
    real=real, fake=fake
  )

def diffusion_recovery_step(energy, base, data, integrator=None,
                            mixing=None, conditional=None, ctx=None):
  data, condition = data.sample(ctx.batch_size)
  loss, args = run_diffusion_recovery_likelihood(
    energy, base, data, condition,
    integrator=integrator, mixing=mixing,
    conditional=conditional
  )
  ctx.argmin(density_ratio_loss=loss)
  return args
