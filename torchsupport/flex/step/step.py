# TODO

class Step:
  def __init__(self, run, ctx=None):
    self.ctx = ctx
    self.name = None
    self.run = run

  def __setattr__(self, name, value):
    if isinstance(value, Step):
      value.ctx = self.ctx
    super().__setattr__(name, value)

  def step(self):
    return self.run(self.ctx)

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

  def step(self):
    with self.update as update:
      result = self.run(self.ctx)
      update.target = 0.0
      if result is not None:
        update.target = result
      update.target += self.ctx.loss
