from torchsupport.new_training.context import ctx
from torchsupport.new_training.tasklets.tasklet import Tasklet

class Parametric:
  def __init__(self, data, run, loss):
    self.data = data
    self._run = run
    self._loss = loss

  def __call__(self, parameters):
    data = self.data.sample(parameters) + ctx(parameters=parameters)
    args = self._run(data)
    loss, context = self._loss(args)
    return ctx(
      loss=loss,
      context=context.detach()
    )

class ParametricObject(Tasklet, Parametric):
  def __init__(self, data):
    super().__init__(data, self.run, self.loss)

  def run(self, data):
    raise NotImplementedError("Abstract.")

  def loss(self, data):
    raise NotImplementedError("Abstract.")

class Serial:
  def __rshift__(self, other):
    return SerialComposition([self, other])

class SerialComposition(Serial):
  def __init__(self, steps):
    self.steps = []
    for item in steps:
      if isinstance(item, SerialComposition):
        self.steps += item.steps
      else:
        self.steps.append(item)

  def __call__(self, parameters):
    history = []
    for step in self.steps:
      history.append(step(parameters))
    result = history[-1]
    return result# + ctx(history=history)

class Step(Serial):
  def __init__(self, parameters, parametric):
    self.parameters = parameters
    self.parametric = parametric

  def __call__(self, parameters):
    self.parameters.init(parameters)
    loss, context = self.parametric(parameters)
    loss.backward()
    self.parameters.step(parameters)
    return ctx(loss=loss, context=context).detach()
