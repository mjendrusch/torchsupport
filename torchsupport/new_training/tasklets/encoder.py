import torch
from torch.distributions import Distribution, kl_divergence

from torchsupport.new_training.tasklets.tasklet import Tasklet

class Encoder(Tasklet):
  def __init__(self, encoder):
    self.encoder = encoder

  def run(self, inputs, args) -> "code":
    code = self.encoder(inputs, args)
    return code

  def loss(self, code) -> []:
    return 0.0

class KLEncoder(Encoder):
  def __init__(self, encoder, prior):
    super().__init__(encoder)
    self.prior = prior
    self.prior_value = None
    if isinstance(prior, Distribution):
      self.prior_value = prior

  def loss(self, code, args) -> "kl_loss":
    if self.prior_value is not None:
      return kl_divergence(code, self.prior_value)
    prior_value = self.prior(args)
    return kl_divergence(code, prior_value)

class LpEncoder(Encoder):
  def __init__(self, encoder, p=2):
    super().__init__(encoder)
    self.p = p

  def loss(self, code, args) -> "lp_loss":
    result = code.view(code.size(0), -1).norm(p=self.p, dim=1)
    result = result.mean()
    return result

class VQEncoder(Encoder):
  def __init__(self, encoder, codebook):
    super().__init__(encoder)
    self.codebook = codebook

  def run(self, inputs, args) -> (
      "code", "code_value", "code_assignment"
  ):
    value = self.encoder(inputs, args)
    code, assignment = self.codebook(value)
    return code, value, assignment

  def loss(self, code_value, code_assignment) -> "commitment_loss":
    result = (code_value - code_assignment).view(code_value.size(0), -1)
    result = result.norm(dim=1)
    return result.mean()
