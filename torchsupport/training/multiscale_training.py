from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import OneHotCategorical

from torchsupport.training.state import (
  NetNameListState, TrainingState
)
from torchsupport.training.multistep_training import MultistepTraining, step_descriptor
from torchsupport.data.io import to_device, make_differentiable, detach
from torchsupport.data.collate import DataLoader, default_collate
from torchsupport.data.match import match, MatchableList
from torchsupport.data.namedtuple import NamedTuple

class MultiscaleNet(nn.Module):
  def __init__(self, nets):
    super().__init__()
    self.scales = nn.ModuleList(nets)

  @property
  def path(self):
    return PathNet(self.scales)

class PathNet(MultiscaleNet):
  def __init__(self, nets):
    super().__init__(nets)

  def forward(self, data):
    inputs, masks = data
    sample = None
    policies = []
    priors = []
    posteriors = []
    tasks = []
    for inp, msk, net in zip(inputs, masks, self.scales):
      inp = inp.transpose(0, 1)
      shape = inp.shape
      # main branch
      task, (prior, sample, policy) = net(inp[0], mask=msk, sample=sample)
      # side branch
      if inp.size(1) > 1:
        inp = inp[1:].reshape(-1, *inp.shape[2:])
        _, (_, s_sample, _) = net(inp, mask=None, sample=None)
        sample = torch.cat((sample, s_sample), dim=0)

      # combine branches
      sample = sample.view(*shape[:2], *sample.shape[1:]).transpose(0, 1)
      policies.append(policy)
      priors.append(prior)
      posteriors.append(sample)
      tasks.append(task)

    policies = policies[1:]
    priors = priors[1:]
    posteriors = posteriors[:-1]

    return tasks, NamedTuple(prior=priors, posterior=posteriors, policy=policies)

class MultiscaleTraining(MultistepTraining):
  step_order = [
    "separate_step", "path_step"
  ]
  def __init__(self, net, separate_data, stack_data, path_data,
               optimizer=None, optimizer_kwargs=None,
               network_options=None, mapping_options=None,
               data_options=None, **kwargs):
    optimizer = optimizer or torch.optim.AdamW
    optimizer_kwargs = optimizer_kwargs or {}
    network_options = network_options or {}
    mapping_options = mapping_options or {}
    data_options = data_options or {}

    self.net = ...
    self.value = ...
    self.target = ...

    networks = dict(
      net=(net, optimizer, optimizer_kwargs),
    )
    networks.update(network_options)

    mapping = dict(
      separate_step=["net"],
      path_step=["net"]
    )
    mapping.update(mapping_options)

    data = dict(
      separate_step=separate_data,
      stack_step=stack_data,
      path_step=path_data
    )
    data.update(data_options)

    super().__init__(networks, mapping, data, **kwargs)

  def run_task(self, net, data):
    raise NotImplementedError("Abstract.")

  def task_loss(self, *args):
    raise NotImplementedError("Abstract.")

  def reward(self, *args):
    raise NotImplementedError("Abstract.")

  def policy_loss(self, *args):
    raise NotImplementedError("Abstract.")

  @step_descriptor(n_steps="n_separate", every="every_separate")
  def separate_step(self, scale_data):
    loss = 0.0
    for idx, (scale, data) in enumerate(zip(self.net.scales, scale_data)):
      task_args, *_ = self.run_task(scale, data)
      level_loss = self.task_loss(*task_args)
      self.current_losses[f"separate level {idx}"] = float(level_loss)
      loss += level_loss
    loss = loss / len(scale_data)
    self.current_losses["separate task"] = float(loss)
    return loss

  def stack_loss(self, task_args, scale_args):
    task_loss = 0.0
    count = 0
    for idx, ta in enumerate(zip(*task_args)):
      level_loss = self.task_loss(*ta)
      self.current_losses[f"level {idx}"] = float(level_loss)
      task_loss += level_loss
      count += 1
    task_loss = task_loss / count
    self.current_losses["task"] = float(task_loss)
    prior_loss = match(
      MatchableList(scale_args.prior),
      MatchableList(detach(scale_args.posterior))
    )
    self.current_losses["prior"] = float(prior_loss)
    policy_loss = self.policy_loss(task_args, scale_args)
    self.current_losses["policy"] = float(policy_loss)
    result = task_loss + 0.001 * prior_loss + 0.001 * policy_loss
    return result

  def run_path(self, data):
    task_args, scale_args = self.run_task(self.net.path, data)
    return task_args, scale_args

  @step_descriptor(n_steps="n_path", every="every_path")
  def path_step(self, data):
    args = self.run_path(data)
    return self.stack_loss(*args)

class MultiscaleClassifierTraining(MultiscaleTraining):
  def run_task(self, net, data):
    data, labels = data
    predictions, scale_args = net(data)
    task_args = (predictions, labels)
    return task_args, scale_args

  def reward(self, task_args, scale_args):
    # curiosity type reward
    result = []
    for prior, posterior in zip(scale_args.prior, scale_args.posterior):
      result.append(-prior.log_prob(posterior).mean(dim=2).detach())
    return result

  def task_loss(self, predictions, labels):
    return func.cross_entropy(predictions, labels)

  def policy_loss(self, task_args, scale_args):
    reward = self.reward(task_args, scale_args)
    nll = 0.0
    for policy, rw in zip(scale_args.policy, reward):
      nll += ((policy - rw) ** 2).mean()
    return nll
