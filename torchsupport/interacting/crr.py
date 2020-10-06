from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.interacting.off_policy_training import OffPolicyTraining

class CRRTraining(OffPolicyTraining):
  def __init__(self, policy, value, agent, environment,
               beta=1.0, clip=None, tau=5e-3, **kwargs):
    self.value = ...
    super().__init__(
      policy, agent, environment,
      {"value": value}, **kwargs
    )
    self.beta = beta
    self.clip = clip
    self.tau = tau
    self.target = deepcopy(value)
    self.policy_target = deepcopy(policy)

  def _update_target(self, target, source):
    if self.step_id % 100 == 1:
      with torch.no_grad():
        tp = target.parameters()
        ap = source.parameters()
        for t, a in zip(tp, ap):
          t -= t
          t += a

  def update_target(self):
    self._update_target(self.target, self.value)
    self._update_target(self.policy_target, self.policy)

  def action_nll(self, policy, action):
    return func.cross_entropy(policy, action, reduction='none')

  def policy_loss(self, policy, action, advantage):
    weight = torch.exp(advantage / self.beta)
    if self.clip is not None:
      weight = weight.clamp(0, self.clip)
    negative_log_likelihood = self.action_nll(policy, action)
    weighted_loss = negative_log_likelihood * weight
    return weighted_loss.mean()

  def state_value(self, state, value=None, policy=None):
    value = value or self.value
    policy = policy or self.policy
    action_value = value(state)
    policy = policy(state)
    pmax = policy.argmax(dim=1)
    ind = torch.arange(
      action_value.size(0),
      dtype=torch.long,
      device=action_value.device
    )
    expected = action_value[ind, pmax]
    return expected

  def run_policy(self, sample):
    initial_state = sample.initial_state
    action = sample.action

    with torch.no_grad():
      action_value = self.value(initial_state)
      inds = torch.arange(action.size(0), device=action.device)
      action_value = action_value[inds, action]
      value = self.state_value(initial_state)
      advantage = action_value - value

    self.current_losses["mean advantage"] = float(advantage.mean())

    policy = self.policy(initial_state)

    return policy, action, advantage

  def auxiliary_loss(self, value, target):
    return func.mse_loss(value.view(-1), target.view(-1))

  def run_auxiliary(self, sample):
    self.update_target()

    initial_state = sample.initial_state
    final_state = sample.final_state
    action = sample.action
    rewards = sample.rewards
    action_value = self.value(initial_state)
    inds = torch.arange(action.size(0), device=action.device)
    action_value = action_value[inds, action]

    with torch.no_grad():
      state_value = self.state_value(
        final_state, value=self.target,
        policy=self.policy_target
      )
      done_mask = 1.0 - sample.done.float()
      target = rewards + self.discount * done_mask * state_value

    self.current_losses["mean state value"] = float(state_value.mean())
    self.current_losses["mean target value"] = float(target.mean())

    return action_value, target
