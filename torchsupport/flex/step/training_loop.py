import time

from torchsupport.flex.step.step import Step
from torchsupport.flex.step.loop import Loop, ConfiguredStep, SequentialStep

class LogStep(Step):
  @staticmethod
  def run_log(ctx=None):
    ctx.logger.log()

  def __init__(self, ctx=None):
    super().__init__(LogStep.run_log, ctx=ctx)

class CheckpointStep(Step):
  @staticmethod
  def run_checkpoint(ctx=None):
    ctx.checkpoint.save()

class TrainingLoop(Loop):
  def __init__(self, num_steps=1000, ctx=None):
    super().__init__(num_steps=num_steps, ctx=ctx)
    self.setup = SequentialStep()
    self.teardown = SequentialStep()

  def log(self):
    for name, value in self.ctx.log_store.items():
      self.ctx.logger.log(name, value, self.ctx.step_id)
    self.ctx.log_store = {}

  def checkpoint(self):
    if self.ctx.step_id % self.ctx.checkpoint_interval == 0:
      self.ctx.checkpoint.checkpoint()

  def save(self):
    if self.ctx.save_time is None:
      self.ctx.save_time = time.monotonic()
    time_since_last_save = time.monotonic() - self.ctx.save_time
    if time_since_last_save > self.ctx.save_interval:
      self.ctx.checkpoint.save()
      self.ctx.save_time = time.monotonic()

  def add(self, every=1, num_steps=1, **kwargs):
    for name, step in kwargs.items():
      step.ctx = self.ctx
      if not isinstance(step, ConfiguredStep):
        step = ConfiguredStep(
          step, every=every,
          num_steps=num_steps,
          ctx=self.ctx
        )
      step.name = name
      setattr(self, name, step)
      self.run.append(step)
    return self

  def step(self):
    self.setup()
    for idx in range(self.ctx.step_id, self.num_steps):
      self.ctx.step_id = idx
      self.setup()
      for step in self.run:
        step()
      self.log()
      self.checkpoint()
      self.save()
    self.teardown()

if __name__ == "__main__":
  loop = TrainingLoop()
  loop.add(first=SequentialStep(), every=10, num_steps=1) \
      .add(second=SequentialStep(), every=1, num_steps=1) \
      .add(third=SequentialStep(), every=5, num_steps=10)
  loop.setup <<= Step(lambda ctx: ctx)
