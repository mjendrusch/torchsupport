import torch
import torch.nn as nn

class Update:
  def __init__(self, parameter_sources, optimizer=None,
               gradient_action=None, **kwargs):
    optimizer = optimizer or torch.optim.Adam
    self.gradient_action = gradient_action
    self.update_actions = []
    self.parameters = []
    for parameter_source in parameter_sources:
      if hasattr(parameter_source, "update_action"):
        self.update_actions.append(parameter_source)
      elif isinstance(parameter_source, nn.Module):
        self.parameters += parameter_source.parameters()
      else:
        self.parameters.append(parameter_source)
    self.optimizer = optimizer(self.parameters, **kwargs)
    self.target = None

  def process_gradients(self):
    if self.gradient_action is not None:
      self.gradient_action(self.parameters)

  def __enter__(self, *args, **kwargs):
    self.optimizer.zero_grad()
    return self

  def __exit__(self, *args, **kwargs):
    loss = self.target
    loss.backward()
    self.process_gradients()
    self.target = None
    self.optimizer.step()
    for update_action in self.update_actions:
      update_action.update_action()
