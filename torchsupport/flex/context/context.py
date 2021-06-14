import torch.nn as nn
from torchsupport.flex.checkpointing.savable import is_savable, Savable
from torchsupport.flex.checkpointing.checkpoint import Checkpoint
from torchsupport.flex.step.training_loop import TrainingLoop
from torchsupport.flex.log.tensorboard_logger import TensorboardLogger

class ContextSwitch:
  def __init__(self, ctx, target):
    self.ctx = ctx
    self.target = target

  def __enter__(self, *args, **kwargs):
    self.ctx.target_stack.append(self.target)
    return self

  def __exit__(self, *args, **kwargs):
    self.ctx.target_stack.pop()

class Context:
  def __init__(self):
    self.target_stack = []
    self.store = ...
    self.log_store = {}

  def switch(self, target):
    return ContextSwitch(self, target)

  def store_path(self, key):
    return ".".join([
      item.name
      for item in self.target_stack
      if item.name is not None
    ] + [key])

  def log(self, **kwargs):
    for key, value in kwargs.items():
      self.log_store[self.store_path(key)] = value

class OptimizationContext(Context):
  def __init__(self):
    super().__init__()
    self._loss = 0.0

  @property
  def loss(self):
    result = self._loss
    self._loss = 0.0
    return result

  @loss.setter
  def loss(self, value):
    self._loss = value
    return value

  def argmin(self, **kwargs):
    for key, value in kwargs.items():
      value = value.mean()
      self.loss += value
      log_value = float(value)
      self.log(**{key: log_value})

  def argmax(self, **kwargs):
    for key, value in kwargs.items():
      value = value.mean()
      self.loss -= value
      log_value = float(value)
      self.log(**{key: log_value})

class RedirectContext(OptimizationContext):
  def __init__(self):
    super().__init__()
    self.loss_store = {}

  @property
  def loss(self):
    result = self.loss_store
    self.loss_store = {}
    return result

  def argmin(self, **kwargs):
    for key, value in kwargs.items():
      value = value.mean()
      self.loss_store[key] = value

  def argmax(self, **kwargs):
    for key, value in kwargs.items():
      value = value.mean()
      self.loss_store[key] = -value

class TrainingContext(OptimizationContext, Savable):
  def __init__(self, path=".",
               max_steps=int(1e7),
               batch_size=128,
               num_workers=8,
               device="cpu",
               verbose=False,
               report_interval=10,
               checkpoint_interval=1000,
               save_interval=600,
               logger=TensorboardLogger):
    super().__init__()
    self.path = path
    self.max_steps = max_steps
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.device = device
    self.verbose = verbose
    self.report_interval = report_interval
    self.checkpoint_interval = checkpoint_interval

    self.checkpoint = Checkpoint(self)
    self.logger = logger(self.path)
    self.loop = TrainingLoop(num_steps=self.max_steps, ctx=self)
    self.loop.name = None
    self.checkpoint.add_save(loop=self.loop)

    self.save_interval = save_interval
    self.save_time = None
    self.step_id = 0

  def train(self):
    self.loop.step()

  def load(self):
    self.checkpoint.load()

  def add(self, *args, **kwargs):
    self.loop.add(*args, **kwargs)

  def try_add_checkpoint(self, name, value):
    if isinstance(value, nn.Module):
      self.checkpoint.add_checkpoint(**{name: value})
    elif is_savable(value):
      self.checkpoint.add_save(**{name: value})

  def register(self, **kwargs):
    for name, value in kwargs.items():
      self.try_add_checkpoint(name, value)
      if hasattr(value, "ctx"):
        value.ctx = self
      setattr(self, name, value)

  def get_step(self, name, base=True, index=0):
    confstep = getattr(self.loop, name)
    if base:
      if index is None:
        index = slice(None)
      step = confstep.run[index]
    else:
      step = confstep
    return step

  def write(self, data, name):
    data[name] = {"step_id": self.step_id}

  def read(self, data, name):
    self.step_id = data[name]["step_id"]
