import torch
import torch.nn as nn
import torch.nn.functional as func

class CriticLoss:
  def __init__(self, ctx=None):
    self.ctx = ctx

  def critic(self, real, fake):
    return 0.0

  def generator(self, real):
    return 0.0

def relativistic(real, fake):
  real = real - fake.mean(dim=0, keepdim=True)
  fake = fake - real.mean(dim=0, keepdim=True)
  return real, fake

class non_saturating(CriticLoss):
  def __init__(self, ctx=None, smoothing=0.0):
    super().__init__(ctx=ctx)
    self.smoothing = smoothing

  def critic(self, real, fake):
    real = func.binary_cross_entropy_with_logits(
      real, torch.zeros_like(real) + self.smoothing
    ).mean()
    fake = func.binary_cross_entropy_with_logits(
      fake, torch.ones_like(fake)
    ).mean()
    return real + fake

  def generator(self, fake):
    return func.binary_cross_entropy_with_logits(
      fake, torch.zeros_like(fake)
    )

class least_squares(CriticLoss):
  def __init__(self, real=1.0, fake=0.0, ctx=None):
    super().__init__(ctx=ctx)
    self.real = real
    self.fake = fake

  def critic(self, real, fake):
    real = ((real - self.real) ** 2).mean()
    fake = ((fake - self.fake) ** 2).mean()
    return real + fake

  def generator(self, fake):
    return ((fake - self.real) ** 2).mean()

class energy_based(CriticLoss):
  def critic(self, real, fake):
    return real.mean() - fake.mean()

  def generator(self, fake):
    return fake.mean()
