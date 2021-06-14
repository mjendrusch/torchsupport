from functools import partial
from torchsupport.flex.step.loop import ConfiguredStep

import torch
from torchsupport.flex.tasks.likelihood.maximum_likelihood import maximum_likelihood_step

from torchsupport.data.io import to_device
from torchsupport.flex.step.step import EvalStep, UpdateStep
from torchsupport.flex.update.update import Update
from torchsupport.flex.context.context import TrainingContext
from torchsupport.flex.utils import filter_kwargs

def conditional_mle_training(model, data, valid_data=None,
                             optimizer=torch.optim.Adam,
                             optimizer_kwargs=None,
                             eval_no_grad=True,
                             **kwargs):
  opt = filter_kwargs(kwargs, ctx=TrainingContext)
  ctx = TrainingContext(**opt.ctx)
  ctx.optimizer = optimizer

  # networks to device
  ctx.register(
    data=to_device(data, ctx.device),
    model=to_device(model, ctx.device)
  )

  ctx.add(train_step=UpdateStep(
      partial(maximum_likelihood_step, ctx.model, ctx.data),
      Update([ctx.model], optimizer=ctx.optimizer, **(optimizer_kwargs or {})),
      ctx=ctx
  ))

  if valid_data is not None:
    ctx.register(valid_data=to_device(valid_data, ctx.device))
    ctx.add(valid_step=EvalStep(
      partial(maximum_likelihood_step, ctx.model, ctx.valid_data),
      modules=[ctx.model], no_grad=eval_no_grad, ctx=ctx
    ), every=ctx.report_interval)
  return ctx
