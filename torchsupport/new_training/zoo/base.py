from torchsupport.new_training.loop.loop import TrainingLoop
from torchsupport.new_training.parametric import Parametric, Step

class TrainingBase:
  def __init__(self, save=None, logger=None, step=None, log=None,
               checkpoint=None, prefix=".", network_path=".",
               batch_size=64, num_workers=8, max_steps=int(1e6),
               report_interval=10, checkpoint_interval=1000, **kwargs):
    self.prefix = prefix
    self.network_path = network_path
    self.full_path = f"{prefix}/{network_path}"
    self.max_steps = max_steps
    self.report_interval = report_interval
    self.checkpoint_interval = checkpoint_interval
    self.batch_size = batch_size
    self.num_workers = num_workers

    self._step = step
    self._checkpoint = checkpoint
    self._save = save(self.full_path)
    self.logger = logger(self.full_path)
    self._log = log(self.logger)

  def set_dependents(self, **kwargs):
    pass

  def step(self):
    return self._step

  def log(self):
    return self._log

  def checkpoint(self):
    return self._checkpoint

  def save(self):
    return self._save

  def load(self):
    self._save.load()

  def loop(self):
    return TrainingLoop(
      self.step(),
      self.log(),
      self.checkpoint(),
      self.save(),
      max_steps=self.max_steps,
      report_interval=self.report_interval,
      checkpoint_interval=self.checkpoint_interval
    )

  def train(self):
    runner = self.loop()
    return runner.run()

class OneStepTraining(TrainingBase):
  def __init__(self, data=None, parameters=None, tasklet=None, **kwargs):
    super().__init__(**kwargs)
    self.set_dependents(
      data=data,
      parameters=parameters,
      tasklet=tasklet,
      **kwargs
    )

  def set_dependents(self, data=None, parameters=None, tasklet=None, **kwargs):
    super().set_dependents(**kwargs)
    self.tasklet = tasklet
    self.parameters = parameters
    self.data = data

  def parametric(self):
    return Parametric(self.data, self.run(), self.loss())

  def step(self):
    return Step(
      self.parameters,
      self.parametric()
    )

  def run(self):
    run = self.tasklet.run if self.tasklet else None
    return run

  def loss(self):
    loss = self.tasklet.loss if self.tasklet else None
    return loss

# class NStepTraining(TrainingBase):
#   def __init__(self, )
