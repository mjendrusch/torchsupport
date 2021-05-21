import torch
import torch.nn as nn

class Update:
  def __init__(self, parameter_sources, optimizer=None, **kwargs):
    optimizer = optimizer or torch.optim.Adam
    self.update_actions = []
    parameters = []
    for parameter_source in parameter_sources:
      if hasattr(parameter_source, "update_action"):
        self.update_actions.append(parameter_source)
      elif isinstance(parameter_source, nn.Module):
        parameters += parameter_source.parameters()
      else:
        parameters.append(parameter_source)
    self.optimizer = optimizer(parameters, **kwargs)
    self.target = None

  def __enter__(self, *args, **kwargs):
    self.optimizer.zero_grad()
    return self

  def __exit__(self, *args, **kwargs):
    loss = self.target
    loss.backward()
    self.target = None
    self.optimizer.step()
    for update_action in self.update_actions:
      update_action.update_action()
