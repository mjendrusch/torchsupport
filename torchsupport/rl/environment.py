import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.rl.trajectory import Experience

class Environment:
  def reset(self):
    raise NotImplementedError

  def action_space(self):
    raise NotImplementedError

  def observation_space(self):
    raise NotImplementedError

  def is_done(self):
    raise NotImplementedError

  def observe(self):
    raise NotImplementedError

  def act(self, action):
    raise NotImplementedError
