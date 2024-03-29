from torchsupport.flex.step.step import Step
from torchsupport.flex.checkpointing.savable import is_savable

class SequentialStep(Step):
  def __init__(self, ctx=None):
    super().__init__(None, ctx=ctx)
    self.run = []

  def add(self, **kwargs):
    for name, step in kwargs.items():
      setattr(self, name, step)
      step.name = name
      self.run.append(step)
    return self

  def write(self, data, name):
    for idx, rval in enumerate(self.run):
      if is_savable(rval):
        rval.write(data, f"{name}.run_{idx}")

  def read(self, data, name):
    for idx, rval in enumerate(self.run):
      if is_savable(rval):
        rval.read(data, f"{name}.run_{idx}")

  def __lshift__(self, other):
    self.add(**{f"step_{len(self.run)}": other})
    return self

  def step(self):
    for step in self.run:
      step()

class Loop(SequentialStep):
  def __init__(self, num_steps=1000, ctx=None):
    super().__init__(ctx=ctx)
    self.num_steps = num_steps

  def step(self):
    for idx in range(self.num_steps):
      super().step()

class ConfiguredStep(Loop):
  def __init__(self, step, every=1, num_steps=1, ctx=None):
    super().__init__(num_steps=num_steps, ctx=ctx)
    self.every = every
    self.add(inner=step)

  def step(self):
    if self.ctx.step_id % self.every == 0:
      super().step()
