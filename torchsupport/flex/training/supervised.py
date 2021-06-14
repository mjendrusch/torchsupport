from functools import partial
from torchsupport.flex.step.loop import ConfiguredStep

import torch
from torchsupport.flex.tasks.likelihood.maximum_likelihood import supervised_step

from torchsupport.data.io import to_device
from torchsupport.flex.step.step import EvalStep, UpdateStep
from torchsupport.flex.update.update import Update
from torchsupport.flex.context.context import TrainingContext
from torchsupport.flex.utils import filter_kwargs

def supervised_training(net, data, valid_data=None, losses=None,
                        optimizer=torch.optim.Adam,
                        optimizer_kwargs=None,
                        eval_no_grad=True,
                        **kwargs):
  opt = filter_kwargs(kwargs, ctx=TrainingContext)
  ctx = TrainingContext(**opt.ctx)
  ctx.optimizer = optimizer
  ctx.losses = losses

  # networks to device
  ctx.register(
    data=to_device(data, ctx.device),
    net=to_device(net, ctx.device)
  )

  ctx.add(train_step=UpdateStep(
      partial(supervised_step, ctx.net, ctx.data, losses=ctx.losses),
      Update([ctx.net], optimizer=ctx.optimizer, **(optimizer_kwargs or {})),
      ctx=ctx
  ))

  if valid_data is not None:
    ctx.register(valid_data=to_device(valid_data, ctx.device))
    ctx.add(valid_step=EvalStep(
      partial(supervised_step, ctx.net, ctx.valid_data, losses=ctx.losses),
      modules=[ctx.net], no_grad=eval_no_grad, ctx=ctx
    ), every=ctx.report_interval)
  return ctx
