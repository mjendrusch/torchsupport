from copy import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.data.io import DeviceMovable, to_device
from torchsupport.data.collate import Collatable, default_collate
from torchsupport.structured.chunkable import Chunkable, scatter_chunked

def _clone_aux(data):
  if isinstance(data, torch.Tensor):
    return data.clone()
  if isinstance(data, (list, tuple)):
    return type(data)([_clone_aux(item) for item in data])
  if isinstance(data, dict):
    return {key: _clone_aux(data[key]) for key in data}
  return data

def _numpy_aux(data):
  if isinstance(data, torch.Tensor):
    return data.numpy()
  if isinstance(data, (list, tuple)):
    return type(data)([_numpy_aux(item) for item in data])
  if isinstance(data, dict):
    return {key: _numpy_aux(data[key]) for key in data}
  return data

def _tensor_aux(data):
  if isinstance(data, np.ndarray):
    return torch.Tensor(data)
  if isinstance(data, (list, tuple)):
    return type(data)([_tensor_aux(item) for item in data])
  if isinstance(data, dict):
    return {key: _tensor_aux(data[key]) for key in data}
  return data

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

  def numpy(self):
    self.initial_state = _numpy_aux(self.initial_state)
    self.final_state = _numpy_aux(self.final_state)
    self.terminal = _numpy_aux(self.terminal)
    self.action = _numpy_aux(self.action)
    self.reward = _numpy_aux(self.reward)
    self.logits = _numpy_aux(self.logits)
    self.outputs = _numpy_aux(self.outputs)
    return self

  def torch(self):
    self.initial_state = _tensor_aux(self.initial_state)
    self.final_state = _tensor_aux(self.final_state)
    self.terminal = self.terminal
    self.action = _tensor_aux(self.action)
    self.reward = _tensor_aux(self.reward)
    self.logits = _tensor_aux(self.logits)
    self.outputs = _tensor_aux(self.outputs)
    return self

  def clone(self):
    return Experience(
      initial_state=_clone_aux(self.initial_state),
      final_state=_clone_aux(self.final_state),
      action=_clone_aux(self.action),
      reward=_clone_aux(self.reward),
      terminal=self.terminal,
      logits=_clone_aux(self.logits),
      outputs=_clone_aux(self.logits)
    )

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

  def numpy(self):
    self.experiences = [
      exp.numpy()
      for exp in self.experiences
    ]
    return self

  def torch(self):
    self.experiences = [
      exp.torch()
      for exp in self.experiences
    ]
    return self

  def clone(self):
    result = copy(self)
    result.experiences = [
      exp.clone()
      for exp in result.experiences
    ]
    return result

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
