from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.rl.trajectory import Experience
from torchsupport.rl.agent import Agent
from torchsupport.rl.off_policy import OffPolicyTraining

class DQNAgent(Agent):
  def __init__(self, agent, tau=0.01, eps=1.0):
    super().__init__()
    self.tau = tau
    self.register_buffer("eps", torch.tensor(eps))
    self.value = agent
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

class DoubleDQNAgent(Agent):
  def __init__(self, agent):
    self.values = nn.ModuleList([
      agent,
      deepcopy(agent)
    ])
    self.step = 0

  def sample(self, logits):
    return self.target.sample(logits)

  @property
  def target(self):
    return self.values[self.step % 2]

  @property
  def value(self):
    return self.values[(self.step + 1) % 2]

  def forward(self, data, inputs=None):
    return self.value(data, inputs=inputs)

  def update(self):
    self.step += 1

class ClippedDQNAgent(Agent):
  def __init__(self, agent):
    self.value_A = agent
    self.value_B = deepcopy(agent)

  def sample(self, logits):
    return self.value_A.sample(logits)

  def forward(self, data, inputs=None):
    return self.value_A(data, inputs=inputs)

  def update(self):
    # switch A and B
    self.value_A, self.value_B = self.value_B, self.value_A

class ValueTraining(OffPolicyTraining):
  def __init__(self, value, environment, agent_kind=None, agent_kwargs=None, **kwargs):
    agent = agent_kind(value, **agent_kwargs)
    super().__init__(agent, environment, **kwargs)

  def target(self, experience):
    raise NotImplementedError("Abstract.")

  def loss(self, value, target):
    result = func.mse_loss(value, target)
    return result

class DQN(OffPolicyTraining):
  def __init__(self, value, environment, tau=0.01, discount=0.99, **kwargs):
    agent = DQNAgent(value, tau=tau)
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

class ClippedDQN(DQN):
  def __init__(self, value, environment, discount=0.9, **kwargs):
    agent = ClippedDQNAgent(value)
    self.discount = discount
    OffPolicyTraining.__init__(agent, environment, **kwargs)

  def target(self, experience):
    with torch.no_grad():
      observation = experience.final_state
      reward = experience.reward
      terminal = experience.terminal

      # maximum value:
      prediction, _ = self.agent.value_A(observation)
      value_A, _ = self.agent.value_A(observation)
      value_B, _ = self.agent.value_B(observation)
      values_A = value_A[
        torch.arange(0, prediction.size(0)),
        prediction.argmax(dim=1)
      ]
      values_B = value_B[
        torch.arange(0, prediction.size(0)),
        prediction.argmax(dim=1)
      ]
      values = min(values_A, values_B)

      return reward + (1 - terminal.float()) * self.discount * values

class DoubleDQN(DQN):
  def __init__(self, value, environment, discount=0.9, **kwargs):
    agent = DoubleDQNAgent(value)
    self.discount = discount
    OffPolicyTraining.__init__(agent, environment, **kwargs)
