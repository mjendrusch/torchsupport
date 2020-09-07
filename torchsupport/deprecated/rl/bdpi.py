from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.rl.trajectory import Experience
from torchsupport.rl.agent import Agent
from torchsupport.rl.off_policy import OffPolicyTraining

class ABCDQNAgent(Agent):
  def __init__(self, actor, critic):
    super().__init__()
    self.value = critic
    self.target = deepcopy(agent)

  def sample(self, logits):
    condition = bool(torch.rand(1)[0] < self.eps)
    if condition:
      logits = torch.rand_like(logits)
    return self.value.sample(logits)

  def forward(self, data, inputs=None):
    return self.value(data)

  def update(self):
    with torch.no_grad():
      tp = self.target.parameters()
      ap = self.value.parameters()
      for t, a in zip(tp, ap):
        t *= (1 - self.tau)
        t += self.tau * a

class BDPITraining(OffPolicyTraining):
  def __init__(self, actor, critic, environment, n_critic=10, discount=0.99, **kwargs):
    agent = ABCDQNAgent(actor, critic, n_critic=n_critic)
    self.discount = discount
    super().__init__(agent, environment, **kwargs)

  def update(self, *data):
    super().update(*data)
    self.agent.update()
    self.anneal_epsilon()

  def target(self, experience):
    with torch.no_grad():
      observation = experience.final_state
      reward = experience.reward
      terminal = experience.terminal

      # maximum value:
      value, _ = self.agent.target(observation)
      prediction, _ = self.agent.value(observation)
      values = value[
        torch.arange(0, prediction.size(0)),
        prediction.argmax(dim=1)
      ]

      return reward + (1 - terminal.float()) * self.discount * values

  def anneal_epsilon(self):
    self.agent.eps = min(torch.tensor(0.1), self.agent.eps - self.step_id * 0.01)

  def run_networks(self, experience):
    observation = experience.initial_state
    action = experience.action
    value, _ = self.agent.value(observation)
    value = value[torch.arange(0, value.size(0)), action]

    target = self.target(experience)
    return value, target

  def loss(self, value, target):
    result = func.mse_loss(value, target)
    return result
