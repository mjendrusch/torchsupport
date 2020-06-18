import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.interacting.off_policy_training import OffPolicyTraining

class AWRTraining(OffPolicyTraining):
  def __init__(self, policy, value, agent, environment,
               beta=1.0, clip=None, **kwargs):
    self.value = ...
    super().__init__(
      policy, agent, environment,
      {"value": value}, **kwargs
    )
    self.beta = beta
    self.clip = clip

  def action_nll(self, policy, action):
    return func.cross_entropy(policy, action, reduction='none')

  def policy_loss(self, policy, action, advantage):
    weight = torch.exp(advantage / self.beta)
    if self.clip is not None:
      weight = weight.clamp(0, self.clip)
    negative_log_likelihood = self.action_nll(policy, action)
    weighted_loss = negative_log_likelihood * weight
    return weighted_loss.mean()

  def run_policy(self, sample):
    initial_state = sample.initial_state
    action = sample.action
    returns = sample.returns

    with torch.no_grad():
      value = self.value(initial_state)
      advantage = returns - value

    policy = self.policy(initial_state)

    return policy, action, advantage

  def auxiliary_loss(self, value, returns):
    return func.mse_loss(value.view(-1), returns.view(-1))

  def run_auxiliary(self, sample):
    initial_state = sample.initial_state
    returns = sample.returns
    value = self.value(initial_state)

    return value, returns
