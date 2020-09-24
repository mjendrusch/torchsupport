import torch
import torch.nn.functional as func

from torchsupport.data.io import make_differentiable

from torchsupport.new_training.tasklets.tasklet import Tasklet
from torchsupport.new_training.tasklets.generator import Generator
from torchsupport.new_training.context import ctx

class DiscriminatorRun(Tasklet):
  def __init__(self, real=True):
    self.real = real

  def run_discriminator(self, discriminator, real, fake, sample):
    real_decision = discriminator(real.inputs, real.args)
    fake_decision = discriminator(fake.inputs, fake.args)
    return real_decision, fake_decision, None

  def run_generator(self, discriminator, real, fake, sample):
    fake_decision = discriminator(fake.inputs, fake.args)
    return None, fake_decision, None

  def run(self, discriminator, real, fake, sample):
    if self.real:
      return self.run_discriminator(discriminator, real, fake, sample)
    else:
      return self.run_generator(discriminator, real, fake, sample)

class RelativisticDiscriminatorRun(DiscriminatorRun):
  def run_discriminator(self, discriminator, real, fake, sample):
    real_decision, fake_decision, _ = super().run_discriminator(
      discriminator, real, fake, sample
    )
    real_decision = real_decision - fake_decision.mean(dim=0, keepdim=True)
    fake_decision = fake_decision - real_decision.mean(dim=0, keepdim=True)
    return real_decision, fake_decision, None

  def run_generator(self, discriminator, real, fake, sample):
    return self.run_discriminator(discriminator, real, fake, sample)

# TODO: UNet discriminator
# class MixDiscriminatorRun(DiscriminatorRun):
#   pass

class DiscriminatorLoss:
  def __init__(self, label_smoothing=0.1, real=True):
    self.real = real
    self.label_smoothing = label_smoothing

  def loss_discriminator(self, real_decision, fake_decision, structure):
    real_score = func.binary_cross_entropy_with_logits(
      real_decision, self.label_smoothing * torch.ones_like(real_decision)
    )
    fake_score = func.binary_cross_entropy_with_logits(
      fake_decision, torch.ones_like(fake_decision)
    )
    return real_score + fake_score

  def loss_generator(self, real_decision, fake_decision, structure):
    if real_decision:
      fake_result = func.binary_cross_entropy_with_logits(
        fake_decision, torch.zeros_like(fake_decision)
      )
      real_result = func.binary_cross_entropy_with_logits(
        real_result, torch.ones_like(real_decision)
      )
      return real_result + fake_result
    return func.binary_cross_entropy_with_logits(
      fake_decision, torch.zeros_like(fake_decision)
    )

  def loss(self, real_decision, fake_decision, structure):
    if self.real:
      return self.loss_discriminator(real_decision, fake_decision, structure)
    else:
      return self.loss_generator(real_decision, fake_decision, structure)

class FisherDiscriminatorLoss(DiscriminatorLoss):
  def activation(self, decision):
    return decision

  def conjugate(self, decision):
    return decision

  def loss_discriminator(self, real_decision, fake_decision, structure):
    real_decision = self.activation(real_decision)
    fake_decision = self.conjugate(self.activation(fake_decision))
    return real_decision - fake_decision

  def loss_generator(self, real_decision, fake_decision, structure):
    fake_decision = self.conjugate(self.activation(fake_decision))
    result = fake_decision
    if real_decision:
      result = fake_decision - self.activation(real_decision)
    return result

class WGANDiscriminatorLoss(FisherDiscriminatorLoss):
  pass

class KLDiscriminatorLoss(FisherDiscriminatorLoss):
  def conjugate(self, decision):
    return (decision - 1).exp()

class ReverseKLDiscriminatorLoss(FisherDiscriminatorLoss):
  def activation(self, decision):
    return -torch.exp(-decision)

  def conjugtate(self, decision):
    return -1 - torch.log(-decision)

class Discriminator(Tasklet):
  def __init__(self, run=None, loss=None, real=True):
    self._run = run or DiscriminatorRun(real=real)
    self._loss = loss or DiscriminatorLoss(real=real)

  def run(self, real, fake, sample) -> (
    "real_decision", "fake_decision", "structure"
  ):
    return self._run(real, fake, sample)

  def loss(self, real_decision, fake_decision, structure) -> (
    "discriminator_loss"
  ):
    return self._loss(real_decision, fake_decision, structure)

# TODO

class GANGenerator(Tasklet):
  def __init__(self, generator, discriminator,
               discriminator_type=Discriminator):
    self.gen = Generator(generator).func
    self.disc = discriminator_type(discriminator).func
    self.noise = generator.sample

  def realness(self, decision):
    return torch.zeros_like(decision)

  def run(self, data, args) -> (
    "real", "fake", "noise", "decision", "realness"
  ):
    sample = self.noise(data, args)
    make_differentiable(sample.noise)
    real = ctx(inputs=data, args=args)
    fake = self.gen.run(sample.noise, sample.args)
    decision = self.disc.run(real, fake, sample)
    realness = self.realness(decision)
    return fake, sample, decision, realness

  def loss(self, decision, realness) -> (
    "generator_loss"
  ):
    return self.disc.loss(decision, realness)

class GANDiscriminator(Tasklet):
  def __init__(self, discriminator, discriminator_type=Discriminator):
    self.disc = discriminator_type(discriminator).func

  def realness(self, decision, real=True):
    return int(real) - torch.ones_like(decision)

  def run(self, inputs, args, fake_inputs, fake_args):
    make_differentiable(inputs)
    make_differentiable(fake_inputs)
    real_decision = self.disc.run(inputs, args)
    fake_decision = self.disc.run(fake_inputs, fake_args)
    real_realness = self.realness(real_decision, real=True)
    fake_realness = self.realness(fake_decision, real=False)
    return real_decision, real_realness, fake_decision, fake_realness

  def loss(self, real_decision, real_realness, fake_decision, fake_realness) -> (
    "discriminator_loss"
  ):
    real_loss = self.disc.loss(real_decision, real_realness)
    fake_loss = self.disc.loss(fake_decision, fake_realness)
    return real_loss + fake_loss

def GAN(generator, discriminator, discriminator_type=Discriminator):
  return (
    GANGenerator(generator, discriminator, discriminator_type=discriminator_type),
    GANDiscriminator(discriminator, discriminator_type=discriminator_type)
  )
