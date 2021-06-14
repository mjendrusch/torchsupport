import torch.nn as nn

class ContextModule(nn.Module):
  def __init__(self, ctx=None):
    super().__init__()
    self._ctx = ctx

  @property
  def ctx(self):
    return self._ctx

  @ctx.setter
  def ctx(self, ctx):
    for module in self.modules():
      if isinstance(module, ContextModule):
        module._ctx = ctx
