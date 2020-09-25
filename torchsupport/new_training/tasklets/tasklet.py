from functools import partial

from .storage import Storage
from torchsupport.data.namedtuple import NamedTuple
from torchsupport.new_training.composable import nominal, Run, Loss, Composable

class Tasklet:
  def __init__(self):
    self.storage = Storage()
    self.parameter_dict = {}

  def __setattr__(self, name, value):
    if isinstance(value, Tasklet):
      value.link_storage(self, name)
    object.__setattr__(self, name, value)

  def store(self, **kwargs):
    for name, data in kwargs.items():
      self.storage[name] = data

  def link_storage(self, target, name):
    target.storage[name] = self.storage

  def parameters(self):
    pass #TODO

  def log(self):
    return NamedTuple(**self.storage)

  def run(self, *args, **kwargs):
    return None

  def loss(self, *args, **kwargs):
    return None

  def step(self, *args, **kwargs):
    return None

class FuncTasklet(Tasklet):
  def __init__(self, run, loss):
    self.run = run
    self.loss = loss
