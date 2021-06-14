import torch
import torch.nn as nn

class Task:
  def parameters(self):
    return []

  def run(self, ctx=None):
    raise NotImplementedError("Abstract.")

  def __call__(self, ctx=None):
    return self.run(ctx=ctx)
