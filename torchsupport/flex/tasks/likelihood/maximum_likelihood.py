from typing import Any, List, NamedTuple, Union

import torch
import torch.nn as nn

from torchsupport.data.match import match
from torchsupport.data.namedtuple import namespace
from torchsupport.flex.context.context_module import ContextModule

class SupervisedLikelihood(nn.Module):
  def __init__(self, predictor):
    super().__init__()
    self.predictor = predictor

  def forward(self, data, condition):
    distribution = self.predictor(condition)
    return distribution.log_prob(data), namespace(distribution=distribution)

def log_likelihood(model, data, condition):
  log_p, args = model(data, condition)
  return log_p, args

def maximum_likelihood_step(model, data, ctx=None):
  data, condition = data.sample(ctx.batch_size)
  log_p, args = model(data, condition)
  ctx.argmax(log_likelihood=log_p)
  return namespace(
    data=data, condition=condition, **args.asdict()
  )

def supervised_loss(prediction, ground_truth, losses):
  loss_value = 0.0
  loss_values = []
  if not isinstance(prediction, (list, tuple)):
    prediction, ground_truth = [prediction], [ground_truth]
  for pred, gt, loss in zip(prediction, ground_truth, losses):
    lval = loss(pred, gt).mean()
    loss_values.append(lval)
    loss_value += lval
  return loss_value, loss_values

class SupervisedArgs(NamedTuple):
  prediction: Union[torch.Tensor, List[torch.Tensor]]
  ground_truth: Union[torch.Tensor, List[torch.Tensor]]
  sample: Union[torch.Tensor, Any]
  losses: List[torch.Tensor]

def run_supervised(model, sample, ground_truth, losses):
  prediction = model(sample)
  loss, losses = supervised_loss(prediction, ground_truth, losses)
  return loss, SupervisedArgs(
    prediction=prediction, ground_truth=ground_truth,
    sample=sample, losses=losses
  )

def supervised_step(model, data, losses=None, ctx=None):
  sample, ground_truth = data.sample(ctx.batch_size)
  loss, args = run_supervised(model, sample, ground_truth, losses)
  ctx.argmin(total_loss=loss)
  for idx, lval in enumerate(args.losses):
    ctx.log(**{f"loss_{idx}": float(lval)})
  return args
