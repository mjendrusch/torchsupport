import torch
from torch.distributions import Distribution, kl_divergence

from torchsupport.data.namedtuple import NamedTuple
from torchsupport.new_training.tasklets.tasklet import Tasklet

class Encoder(Tasklet):
  def __init__(self, encoder):
    super().__init__()
    self.encoder = encoder

  def run(self, inputs, args):
    code = self.encoder(inputs, args)
    self.store(code=code)
    return code

  def loss(self, code, args):
    return 0.0

  def step(self, inputs, args):
    code = self.run(inputs, args)
    return self.loss(code, args)

class KLEncoder(Encoder):
  def __init__(self, encoder, prior):
    super().__init__(encoder)
    self.prior = prior
    self.prior_value = None
    if isinstance(prior, Distribution):
      self.prior_value = prior

  def loss(self, code, args):
    if self.prior_value is not None:
      return kl_divergence(code, self.prior_value)
    prior_value = self.prior(args)
    loss = kl_divergence(code, prior_value)
    self.store(kl_loss=loss)
    return loss

class LpEncoder(Encoder):
  def __init__(self, encoder, p=2):
    super().__init__(encoder)
    self.p = p

  def loss(self, code, args):
    result = code.view(code.size(0), -1).norm(p=self.p, dim=1)
    result = result.mean()
    self.store(lp_loss=result)
    return result

class VQEncoder(Encoder):
  def __init__(self, encoder, codebook):
    super().__init__(encoder)
    self.codebook = codebook

  def run(self, inputs, args):
    value = self.encoder(inputs, args)
    code, assignment = self.codebook(value)
    self.store(value=value, code=code)
    return NamedTuple(
      code=code, value=value, assignment=assignment
    )

  def loss(self, code_value, code_assignment):
    result = (code_value - code_assignment).view(code_value.size(0), -1)
    result = result.norm(dim=1)
    self.store(commitment_loss=result)
    return result.mean()

  def step(self, inputs, args):
    code = self.run(inputs, args)
    loss = self.loss(code.value, code.assignment)
    return loss
