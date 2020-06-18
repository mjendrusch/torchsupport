from copy import copy, deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as func

class Policy(nn.Module):
  def schema(self):
    raise NotImplementedError("Abstract.")

  def move(self):
    return self

  def push(self):
    raise NotImplementedError("Abstract.")

  def pull(self):
    raise NotImplementedError("Abstract.")

  def forward(self, state, inputs=None):
    raise NotImplementedError("Abstract.")

class ModulePolicy(Policy):
  def __init__(self, policy, device="cpu"):
    super().__init__()
    self.policy = policy
    self.device = device

  def move(self):
    result = deepcopy(self)
    result.policy = self.policy.clone_to(self.device)
    return result

  def push(self):
    self.policy.push()

  def pull(self):
    self.policy.pull()

  def schema(self):
    return self.policy.schema()
