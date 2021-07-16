from functools import partial

import torch
from torchsupport.flex.tasks.likelihood.maximum_likelihood import maximum_likelihood_step
from torchsupport.flex.tasks.energy.density_ratio import tdre_step, density_ratio_step

from torchsupport.data.io import to_device
from torchsupport.flex.step.step import UpdateStep
from torchsupport.flex.update.update import Update
from torchsupport.flex.context.context import TrainingContext
from torchsupport.flex.utils import filter_kwargs

def base_dre_training(energy, base, data, train_base=True,
                      base_step=maximum_likelihood_step,
                      optimizer=torch.optim.Adam,
                      base_optimizer_kwargs=None,
                      **kwargs):
  opt = filter_kwargs(kwargs, ctx=TrainingContext)
  ctx = TrainingContext(**opt.ctx)
  ctx.optimizer = optimizer

  # networks to device
  ctx.register(
    data=to_device(data, ctx.device),
    base=to_device(base, ctx.device),
    energy=to_device(energy, ctx.device)
  )

  if train_base:
    ctx.add(base_step=UpdateStep(
        partial(base_step, ctx.base, ctx.data),
        Update([ctx.base], optimizer=ctx.optimizer, **(base_optimizer_kwargs or {})),
        ctx=ctx
    ))
  return ctx

def telescoping_density_ratio_training(energy, base, data, mixing=None,
                                       optimizer_kwargs=None,
                                       telescoping_step=tdre_step,
                                       verbose=True,
                                       **kwargs):
  opt = filter_kwargs(kwargs, ctx=base_dre_training)
  ctx = base_dre_training(energy, base, data, **opt.ctx)

  ctx.add(tdre_step=UpdateStep(
    partial(telescoping_step, ctx.energy, ctx.base, ctx.data, mixing=mixing, verbose=verbose),
    Update([ctx.energy], optimizer=ctx.optimizer, **(optimizer_kwargs or {})),
    ctx=ctx
  ))

  return ctx

def density_ratio_training(energy, base, data, optimizer_kwargs=None,
                           density_ratio_step=density_ratio_step,
                           **kwargs):
  opt = filter_kwargs(kwargs, ctx=base_dre_training)
  ctx = base_dre_training(energy, base, data, **opt.ctx)

  ctx.add(dre_step=UpdateStep(
    partial(density_ratio_step, ctx.energy, ctx.base, ctx.data),
    Update([ctx.energy], optimizer=ctx.optimizer, **(optimizer_kwargs or {})),
    ctx=ctx
  ))

  return ctx
