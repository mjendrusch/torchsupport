from torchsupport.new_training.context import ctx
from torchsupport.new_training.parametric import Serial

class Loop(Serial):
  def __init__(self, step, step_name=None, max_steps=10):
    self.step = step
    self.max_steps = max_steps
    self.step_name = step_name or "step_id"

  def run(self, external_parameters=None):
    context = None
    external_parameters = external_parameters or ctx()
    for step_id in range(self.max_steps):
      parameters = ctx(**{self.step_name: step_id})
      parameters += external_parameters
      context = self.step(parameters)
    return context

  def __call__(self, external_parameters=None):
    return self.run(external_parameters=external_parameters)

class TrainingLoop(Loop):
  def __init__(self, step, log, checkpoint, save,
               max_steps=int(1e6), report_interval=10,
               checkpoint_interval=1000):
    super().__init__(step, max_steps=max_steps)
    self.log = log
    self.checkpoint = checkpoint
    self.report_interval = report_interval
    self.checkpoint_interval = checkpoint_interval
    self.save = save

  def run(self):
    for step_id in range(self.max_steps):
      parameters = ctx(step_id=step_id)
      context = self.step(parameters)
      if step_id % self.report_interval == 0:
        self.log(context)
      if step_id % self.checkpoint_interval == 0:
        self.checkpoint(context)
      self.save(context)

  def __call__(self):
    return self.run()
