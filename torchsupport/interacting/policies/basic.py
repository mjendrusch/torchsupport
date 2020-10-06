import random

import torch
from torch.distributions import Categorical

from torchsupport.data.namedtuple import namedtuple, NamedTuple

from torchsupport.interacting.policies.policy import Policy, ModulePolicy

class RandomPolicy(Policy):
  data_type = namedtuple("Data", ["logits", "outputs"])
  def __init__(self, actions):
    super().__init__()
    self.logits = torch.ones(actions) / actions

  def push(self):
    pass

  def pull(self):
    pass

  def schema(self):
    return self.data_type(
      logits=self.logits, outputs=None
    )

  def forward(self, state, hidden=None):
    action = Categorical(logits=self.logits).sample()
    return action, self.logits, hidden

class CategoricalPolicy(ModulePolicy):
  def forward(self, state, hidden=None):
    if isinstance(state, (list, tuple, NamedTuple)):
      state = [
        item.unsqueeze(0)
        for item in state
      ]
    else:
      state = state.unsqueeze(0)
    hidden = hidden.unsqueeze(0) if hidden else None
    logits = self.policy(
      state, hidden=hidden
    )
    outputs = [None]
    if isinstance(logits, tuple):
      logits, outputs = logits
    action = Categorical(logits=logits).sample()

    return action[0], logits[0], outputs[0]

class EpsilonGreedyPolicy(ModulePolicy):
  def __init__(self, policy, epsilon=0.1):
    super().__init__(policy)
    self.epsilon = epsilon

  def forward(self, state, hidden=None):
    explore = random.random() < self.epsilon
    state = state.unsqueeze(0)
    hidden = hidden.unsqueeze(0) if hidden else None
    logits = self.policy(
      state, hidden=hidden
    )
    outputs = [None]
    if isinstance(logits, tuple):
      logits, outputs = logits

    action = logits.argmax(dim=1)

    if explore:
      logits = torch.ones_like(logits)
      logits = logits / logits.size(1)
      action = Categorical(logits=logits).sample()

    return action[0], logits[0], outputs[0]

class CategoricalGreedyPolicy(EpsilonGreedyPolicy):
  def forward(self, state, hidden=None):
    explore = random.random() < self.epsilon
    state = state.unsqueeze(0)
    hidden = hidden.unsqueeze(0) if hidden else None
    logits = self.policy(
      state, hidden=hidden
    )
    outputs = [None]
    if isinstance(logits, tuple):
      logits, outputs = logits
    action = logits.argmax(dim=1)
    if explore:
      action = Categorical(logits=logits).sample()

    return action[0], logits[0], outputs[0]
