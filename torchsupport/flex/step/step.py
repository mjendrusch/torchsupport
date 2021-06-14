import torch
from torchsupport.flex.checkpointing.savable import Savable, is_savable

class Step(Savable):
  def __init__(self, run, ctx=None):
    self.ctx = ctx
    self.name = None
    self.run = run

  def __setattr__(self, name, value):
    if isinstance(value, Step):
      value.ctx = self.ctx
    super().__setattr__(name, value)

  def extend(self, run):
    crun = self.run
    self.run = lambda ctx=None: run(crun(ctx=ctx), ctx=ctx)

  def step(self):
    return self.run(ctx=self.ctx)

  def write(self, data, name):
    if is_savable(self.run):
      self.run.write(data, f"{name}.run")

  def read(self, data, name):
    if is_savable(self.run):
      self.run.read(data, f"{name}.run")

  def __call__(self):
    with self.ctx.switch(self):
      return self.step()

class EmptyStep(Step):
  @staticmethod
  def noop(ctx):
    return

  def __init__(self, ctx=None):
    super().__init__(run=EmptyStep.noop, ctx=ctx)

class UpdateStep(Step):
  def __init__(self, run, update, ctx=None):
    super().__init__(run, ctx=ctx)
    self.update = update

  def write(self, data, name):
    super().write(data, name)
    if is_savable(self.update):
      self.update.write(data, f"{name}.update")

  def read(self, data, name):
    super().read(data, name)
    if is_savable(self.update):
      self.update.read(data, f"{name}.update")

  def step(self):
    with self.update as update:
      result = self.run(ctx=self.ctx)
      update.target = self.ctx.loss

class EvalStep(Step):
  def __init__(self, run, modules=None,
               no_grad=True, ctx=None):
    super().__init__(run, ctx=ctx)
    self.modules = modules
    self.no_grad = no_grad

  def step(self):
    for net in self.modules:
      net.eval()
    if self.no_grad:
      with torch.no_grad():
        self.run(ctx=self.ctx)
    else:
      self.run(ctx=self.ctx)
    for net in self.modules:
      net.train()
