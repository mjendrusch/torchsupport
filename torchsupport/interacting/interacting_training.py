import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.training.training import Training

class InteractingTraining(Training):
  def __init__(self, tasks, valid_tasks=None,
               max_steps=1_000_000,
               checkpoint_interval=10,
               network_name="network",
               path_prefix=".",
               report_interval=1000,
               verbose=True):
    self.max_steps = max_steps
    self.checkpoint_interval = checkpoint_interval
    self.report_interval = report_interval
    self.step_id = 0
    self.verbose = verbose
    self.tasks = tasks
    self.valid_tasks = valid_tasks or []
    self.checkpoint_path = f"{path_prefix}/{network_name}-checkpoint"
    for task in self.tasks:
      task.register_training(self)

  def step(self):
    for task in self.tasks:
      task_data = task.sample()
      task.step(task_data)
    self.each_step()

  def checkpoint(self):
    for task in self.tasks:
      task.checkpoint()
    self.each_checkpoint()

  def validate(self):
    for task in self.valid_tasks:
      task.valid_step()

  def train(self):
    for task in self.tasks:
      task.initialize()
    for _ in range(self.max_steps):
      self.step()
      if self.step_id % self.report_interval == 0:
        self.validate()
      if self.step_id % self.checkpoint_interval == 0:
        self.checkpoint()
      self.step_id += 1

    task_trainables = {
      task.name: task.trainables
      for task in self.tasks
    }

    return task_trainables
