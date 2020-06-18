import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.data.io import to_device
from torchsupport.interacting.off_policy_training import OffPolicyTraining

class ClonedValue(nn.Module):
  def __init__(self, value, clones=16):
    super().__init__()
    self.clones = nn.ModuleList([
      nn.ModuleList([
        deepcopy(value),
        deepcopy(value)
      ])
      for _ in range(clones)
    ])
    self.keys = [
      0
      for _ in range(clones)
    ]

  def swap(self, position):
    self.keys[position] = not self.keys[position]

  def position(self):
    return random.choice(range(len(self.clones)))

  def forward(self, position, key, *args, **kwargs):
    value = self.clones[position][int(key)]
    return value(*args, **kwargs)

class BDPITraining(OffPolicyTraining):
  def __init__(self, policy, value, agent, environment,
               clones=16, value_lr=0.2, policy_lr=0.05,
               gradient_updates=5, critic_updates=5,
               **kwargs):
    self.value = ...
    super().__init__(
      policy, agent, environment,
      {"value": ClonedValue(value, clones=clones)}, **kwargs
    )
    self.clones = clones
    self.value_lr = value_lr
    self.policy_lr = policy_lr
    self.gradient_updates = gradient_updates
    self.critic_updates = critic_updates
    self.current_position = 0

  def action_nll(self, policy, target):
    #result = func.mse_loss(policy, target, reduction="none").sum(dim=1)
    fw = func.kl_div(policy.log(), target, reduction='none')
    rv = func.kl_div(target.log(), policy, reduction='none')
    result = fw + rv
    return result.sum(dim=1)

  def policy_loss(self, policy, target):
    result = self.action_nll(policy, target)
    return result.mean()

  def run_policy(self, sample, target=None):
    initial_state = sample.initial_state

    logits = self.policy(initial_state)
    policy = logits.softmax(dim=1)

    if target is None:
      position = self.current_position
      key = self.value.keys[position]

      with torch.no_grad():
        value = self.value(position, key, initial_state)

      greedy = value.argmax(dim=1)
      ind = torch.arange(greedy.size(0), device=greedy.device)

      target = torch.zeros_like(value)
      target[ind, greedy] = 1

      target = self.policy_lr * target + (1 - self.policy_lr) * policy
      #target = target.log()

      return policy, target.detach()
    else:
      return policy

  def auxiliary_loss(self, value, target):
    return func.mse_loss(value, target)

  def run_auxiliary(self, sample, target=None):
    initial_state = sample.initial_state
    final_state = sample.final_state
    rewards = sample.rewards
    action = sample.action

    position = self.current_position
    if target is None:
      self.value.swap(position)
    key = self.value.keys[position]

    ind = torch.arange(action.size(0), device=action.device)
    current_q = self.value(position, key, initial_state)[ind, action]

    if target is None:
      with torch.no_grad():
        next_q_a = self.value(position, key, final_state)
        next_q_b = self.value(position, not key, final_state)

        next_greedy_action = next_q_a.argmax(dim=1)
        max_q_a = next_q_a[ind, next_greedy_action]
        max_q_b = next_q_b[ind, next_greedy_action]
        next_q = torch.min(max_q_a, max_q_b)

        done_mask = 1.0 - sample.done.float()

        update = rewards + self.discount * done_mask * next_q - current_q
        target = current_q + self.value_lr * update
      return current_q, target.detach()
    else:
      return current_q

  def policy_step(self):
    data = self.buffer.sample(self.batch_size)
    data = to_device(data, self.device)

    policy, target = self.run_policy(data)

    for _ in range(self.gradient_updates):
      self.optimizer.zero_grad()
      policy = self.run_policy(data, target=target)
      loss = self.policy_loss(policy, target.detach())
      loss.backward()
      self.optimizer.step()

    self.current_losses["policy"] = float(loss)

    self.agent.push()

  def auxiliary_step(self):
    self.current_position = self.value.position()
    for idx in range(self.critic_updates):
      data = self.buffer.sample(self.batch_size)
      data = to_device(data, self.device)

      value, target = self.run_auxiliary(data)
      for _ in range(self.gradient_updates):
        self.auxiliary_optimizer.zero_grad()
        value = self.run_auxiliary(data, target=target)
        loss = self.auxiliary_loss(value, target)
        loss.backward()
        self.auxiliary_optimizer.step()

    self.current_losses["auxiliary"] = float(loss)
