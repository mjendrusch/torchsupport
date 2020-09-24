# from torchsupport.new_training.parameters.update_module import UpdateModule

class Parameters:
  def init(self, *args, **kwargs):
    raise NotImplementedError("Abstract.")

  def step(self, *args, **kwargs):
    raise NotImplementedError("Abstract.")

def _nop(x, y):
  return x

class GradientParameters:
  def __init__(self, modules, optimiser, update=None, **kwargs):
    self.modules = modules
    self.update = update or _nop
    self.optimiser = optimiser(modules.parameters(), **kwargs)

  def init(self, parameters):
    self.optimiser.zero_grad()
    self.update(self.optimiser, parameters)

  def step(self, parameters):
    self.optimiser.step()

class EMAParameters:
  def __init__(self, modules, providers):
    self.modules = modules
    self.providers = providers

  def init(self, parameters):
    pass

  def step(self, parameters):
    pass # TODO

# class UpdateParameters:
#   def __init__(self, modules):
#     self.modules = modules
#     self.parameters = [

#     ]
