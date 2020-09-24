from functools import partial

from torchsupport.new_training.composable import nominal, Run, Loss, Composable

class TaskletMeta(type):
  def __new__(cls, name, bases, attr):
    if "run" in attr:
      attr["run_impl"] = attr["run"]
      attr["run"] = nominal(Run)(attr["run"])
    if "loss" in attr:
      attr["loss_impl"] = attr["loss"]
      attr["loss"] = nominal(Loss)(attr["loss"])
    return super(TaskletMeta, cls).__new__(cls, name, bases, attr)

class StructuralTasklet:
  def __init__(self, run, loss):
    self.run = run.function
    self.loss = loss.function

class Tasklet(metaclass=TaskletMeta):
  @property
  def func(self):
    return StructuralTasklet(self.run, self.loss)

  def run(self, ctx) -> []:
    return None

  def loss(self, ctx) -> []:
    return None

class FuncTasklet(Tasklet):
  def __init__(self, run, loss):
    self.run = run
    self.loss = loss

def tasklet_impl(function, *args, **kwargs):
  return FuncTasklet(*function(*args, **kwargs))

def tasklet(function):
  return partial(tasklet_impl, function)
