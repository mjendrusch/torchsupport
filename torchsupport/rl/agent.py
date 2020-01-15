import torch
import torch.nn as nn
import torch.nn.functional as func

class Agent(nn.Module):
  def sample(self, logits):
    raise NotImplementedError

  def forward(self, state, inputs=None):
    raise NotImplementedError

class MultiAgent(Agent):
  def __init__(self, agents):
    super().__init__()
    self.agents = agents

  def sample(self, logits):
    actions = []
    for idx, agent in enumerate(self.agents):
      if logits[idx] is None:
        actions.append(None)
      else:
        actions.append(agent.sample(logits[idx]))
    return actions

  def forward(self, state, inputs=None):
    inputs = inputs or [None for _ in self.agents]
    logits = []
    outputs = []
    for idx, agent in enumerate(self.agents):
      if state[idx] is None:
        logits.append(None)
        outputs.append(inputs[idx])
      else:
        agent_logits, agent_outputs = agent(state[idx], inputs=inputs[idx])
        logits.append(agent_logits)
        outputs.append(agent_outputs)
    return logits, outputs
