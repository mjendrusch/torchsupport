import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.data.io import DeviceMovable, to_device
from torchsupport.data.collate import Collatable, default_collate
from torchsupport.structured.chunkable import Chunkable, scatter_chunked

class Experience(Collatable, Chunkable, DeviceMovable):
  __slots__ = [
    "initial_state", "action", "reward", "final_state", "terminal", "logits", "outputs"
  ]
  def __init__(self, initial_state, final_state, action, reward,
               terminal=False, logits=None, outputs=None):
    self.initial_state = initial_state
    self.final_state = final_state
    self.terminal = terminal
    self.action = action
    self.reward = reward
    self.logits = logits
    self.outputs = outputs

  @classmethod
  def collate(cls, inputs):
    initial_state = []
    final_state = []
    action = []
    reward = []
    logits = []
    outputs = []
    terminal = []
    for experience in inputs:
      initial_state.append(experience.initial_state)
      final_state.append(experience.final_state)
      action.append(experience.action)
      reward.append(experience.reward)
      logits.append(experience.logits)
      outputs.append(experience.outputs)
      terminal.append(experience.terminal)
    if logits[0] is None:
      logits = None
    else:
      logits = default_collate(logits)
    if outputs[0] is None:
      outputs = None
    else:
      outputs = default_collate(outputs)
    initial_state = default_collate(initial_state)
    final_state = default_collate(final_state)
    action = default_collate(action)
    reward = default_collate(reward)
    terminal = default_collate(terminal)
    result = Experience(
      initial_state=initial_state,
      final_state=final_state,
      action=action,
      reward=reward.float(),
      terminal=terminal,
      logits=logits,
      outputs=outputs
    )
    return result

  def move_to(self, device):
    return Experience(
      initial_state=to_device(self.initial_state, device),
      final_state=to_device(self.final_state, device),
      action=to_device(self.action, device),
      reward=to_device(self.reward, device),
      terminal=to_device(self.terminal, device),
      logits=to_device(self.logits, device),
      outputs=to_device(self.outputs, device)
    )

  def chunk(self, targets):
    initial_state = scatter_chunked(self.initial_state, targets)
    final_state = scatter_chunked(self.final_state, targets)
    action = scatter_chunked(self.action, targets)
    reward = scatter_chunked(self.reward, targets)
    terminal = scatter_chunked(self.terminal, targets)

    nones = [None] * len(targets)
    logits = nones if self.logits is None else scatter_chunked(self.logits, targets)
    outputs = nones if self.outputs is None else scatter_chunked(self.outputs, targets)
    result = [
      Experience(*combination)
      for combination in zip(
        initial_state, final_state, action, reward, terminal, logits, outputs
      )
    ]
    return result

class Trajectory:
  def __init__(self):
    self.experiences = []

  def append(self, experience):
    self.experiences.append(experience)

  def __getitem__(self, index):
    result = ...
    if isinstance(index, slice):
      result = default_collate(self.experiences[index])
    else:
      result = self.experiences[index]
    return result

  def __len__(self):
    return len(self.experiences)

  def complete(self):
    pass

class DiscountedTrajectory(Trajectory):
  def __init__(self, discount=0.99):
    super().__init__()
    self.discount = discount

  @classmethod
  def with_discount(cls, discount):
    return lambda: cls(discount=discount)

  def complete(self):
    previous = 0.0
    for experience in reversed(self.experiences):
      experience.reward += self.discount * previous
      previous = experience.reward
