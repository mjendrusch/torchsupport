from copy import deepcopy
from functools import partial
from torchsupport.flex.tasks.energy.score_matching import linear_noise

import torch
from torchsupport.flex.tasks.energy.relaxed_score_matching import (
  NormalNoise, relaxed_score_matching_step, cnce_step, recovery_likelihood_step
)

from torchsupport.data.io import to_device
from torchsupport.flex.step.step import UpdateStep
from torchsupport.flex.update.update import Update
from torchsupport.flex.context.context import TrainingContext
from torchsupport.flex.utils import filter_kwargs

def base_sm_training(energy, data,
                     optimizer=torch.optim.Adam,
                     **kwargs):
  opt = filter_kwargs(kwargs, ctx=TrainingContext)
  ctx = TrainingContext(**opt.ctx)
  ctx.optimizer = optimizer

  # networks to device
  energy_target = deepcopy(energy)
  ctx.register(
    data=to_device(data, ctx.device),
    energy=to_device(energy, ctx.device),
    energy_target=to_device(energy_target, ctx.device)
  )

  return ctx

def _ema_step(ema_weight=0.999, ctx=None):
  with torch.no_grad():
    for target, source in zip(ctx.energy_target.parameters(),
                              ctx.energy.parameters()):
      target *= ema_weight
      target += source * (1 - ema_weight)

def relaxed_score_matching_training(energy, data,
                                    optimizer_kwargs=None,
                                    score_matching_step=relaxed_score_matching_step,
                                    level_weight=1.0,
                                    level_distribution=NormalNoise(linear_noise(1e-3, 10.0)),
                                    noise_distribution=NormalNoise(1e-3),
                                    loss_scale=1000,
                                    ema_weight=0.999,
                                    **kwargs):
  opt = filter_kwargs(kwargs, ctx=base_sm_training)
  ctx = base_sm_training(energy, data, **opt.ctx)

  ctx.add(tdre_step=UpdateStep(
    partial(
      score_matching_step, ctx.energy, ctx.data,
      level_weight=level_weight,
      level_distribution=level_distribution,
      noise_distribution=noise_distribution,
      loss_scale=loss_scale
    ),
    Update([ctx.energy], optimizer=ctx.optimizer, **(optimizer_kwargs or {})),
    ctx=ctx
  ))
  ctx.add(ema_step=partial(_ema_step, ema_weight=ema_weight, ctx=ctx))

  return ctx

def cnce_training(energy, data,
                  optimizer_kwargs=None,
                  level_distribution=NormalNoise(linear_noise(1e-3, 10.0)),
                  noise_distribution=NormalNoise(1e-3),
                  ema_weight=0.999,
                  **kwargs):
  opt = filter_kwargs(kwargs, ctx=base_sm_training)
  ctx = base_sm_training(energy, data, **opt.ctx)

  ctx.add(tdre_step=UpdateStep(
    partial(
      cnce_step, ctx.energy, ctx.data,
      level_distribution=level_distribution,
      noise_distribution=noise_distribution,
    ),
    Update([ctx.energy], optimizer=ctx.optimizer, **(optimizer_kwargs or {})),
    ctx=ctx
  ))
  ctx.add(ema_step=partial(_ema_step, ema_weight=ema_weight, ctx=ctx))

  return ctx

def recovery_likelihood_training(energy, data,
                                 optimizer_kwargs=None,
                                 level_weight=1.0,
                                 loss_scale=1.0,
                                 level_distribution=NormalNoise(linear_noise(1e-3, 10.0)),
                                 noise_distribution=NormalNoise(1e-3),
                                 ema_weight=0.999,
                                 **kwargs):
  opt = filter_kwargs(kwargs, ctx=base_sm_training)
  ctx = base_sm_training(energy, data, **opt.ctx)

  ctx.add(recovery_likelihood_step=UpdateStep(
    partial(
      recovery_likelihood_step, ctx.energy, ctx.data,
      level_weight=level_weight,
      loss_scale=loss_scale,
      level_distribution=level_distribution,
      noise_distribution=noise_distribution,
    ),
    Update([ctx.energy], optimizer=ctx.optimizer, **(optimizer_kwargs or {})),
    ctx=ctx
  ))
  ctx.add(ema_step=partial(_ema_step, ema_weight=ema_weight, ctx=ctx))

  return ctx
