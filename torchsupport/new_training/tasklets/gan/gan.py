import torch
import torch.nn.functional as func

from torchsupport.data.io import make_differentiable, to_device

from torchsupport.data.namedtuple import NamedTuple, namedtuple
from torchsupport.new_training.tasklets.tasklet import Tasklet
from torchsupport.new_training.tasklets.generator import Generator

DiscriminatorInputs = namedtuple("DiscriminatorInputs", ["data", "args"])
DiscriminatorResult = namedtuple("DiscriminatorResult", ["fake", "real"])

class DiscriminatorRun(Tasklet):
  def __init__(self, real=True):
    super().__init__()
    self.real = real

  def run_discriminator(self, discriminator, fake, real):
    real_decision = discriminator(real.data, real.args)
    fake_decision = discriminator(fake.data, fake.args)
    return DiscriminatorResult(
      fake=fake_decision,
      real=real_decision
    )

  def run_generator(self, discriminator, fake, real=None):
    fake_decision = discriminator(fake.data, fake.args)
    return DiscriminatorResult(
      fake=fake_decision,
      real=None
    )

  def run(self, discriminator, fake, real=None):
    if self.real:
      return self.run_discriminator(discriminator, fake, real)
    else:
      return self.run_generator(discriminator, fake, real)

class RelativisticDiscriminatorRun(DiscriminatorRun):
  def __init__(self, real=True):
    super().__init__(real=real)
    self.real_decision = torch.tensor(0.0, dtype=torch.float)
    self.real_mean = torch.tensor(0.0, dtype=torch.float)

  def save_real(self, real_decision, real_mean):
    self.real_decision = real_decision.cpu().detach()
    self.real_mean = real_mean.cpu().detach()

  def restore_real(self):
    return NamedTuple(
      decision=self.real_decision,
      mean=self.real_mean
    )

  def run_discriminator(self, discriminator, fake, real):
    decision = super().run_discriminator(
      discriminator, fake, real
    )
    fake_mean = decision.fake.mean(dim=0, keepdim=True)
    real_mean = decision.real.mean(dim=0, keepdim=True)
    self.save_real(decision.real, real_mean)
    self.store(
      fake_mean=fake_mean,
      real_mean=real_mean
    )
    real_decision = decision.real - fake_mean
    fake_decision = decision.fake - real_mean
    return DiscriminatorResult(
      fake=fake_decision,
      real=real_decision
    )

  def run_generator(self, discriminator, fake, real=None):
    if real:
      return self.run_discriminator(discriminator, fake, real)
    else:
      fake_decision = discriminator(fake.data, fake.args)
      fake_mean = fake_decision.mean(dim=0, keepdim=True)
      real_decision, real_mean = to_device(
        self.restore_real(), fake_mean.device
      )
      fake_decision = fake_decision - real_mean
      real_decision = real_decision - fake_mean
      return DiscriminatorResult(
        fake=fake_decision,
        real=real_decision
      )

# TODO: UNet discriminator
# class MixDiscriminatorRun(DiscriminatorRun):
#   pass

class DiscriminatorLoss(Tasklet):
  def __init__(self, label_smoothing=0.1, real=True):
    super().__init__()
    self.real = real
    self.label_smoothing = label_smoothing

  def loss_discriminator(self, fake_decision, real_decision):
    real_score = func.binary_cross_entropy_with_logits(
      real_decision, self.label_smoothing * torch.ones_like(real_decision)
    )
    fake_score = func.binary_cross_entropy_with_logits(
      fake_decision, torch.ones_like(fake_decision)
    )
    return real_score + fake_score

  def loss_generator(self, fake_decision, real_decision=None):
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

  def loss(self, fake_decision, real_decision=None):
    if self.real:
      return self.loss_discriminator(fake_decision, real_decision)
    else:
      return self.loss_generator(fake_decision, real_decision)

class FisherDiscriminatorLoss(DiscriminatorLoss):
  def activation(self, decision):
    return decision

  def conjugate(self, decision):
    return decision

  def loss_discriminator(self, fake_decision, real_decision):
    real_decision = self.activation(real_decision)
    fake_decision = self.conjugate(self.activation(fake_decision))
    return real_decision - fake_decision

  def loss_generator(self, fake_decision, real_decision=None):
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
  def __init__(self, discriminator, run=None, loss=None, real=True):
    super().__init__()
    self.discriminator = discriminator
    self._run = run or DiscriminatorRun(real=real)
    self._loss = loss or DiscriminatorLoss(real=real)

  def run(self, fake, real=None):
    return self._run(self.discriminator, fake, real)

  def loss(self, fake_decision, real_decision=None):
    return self._loss(fake_decision, real_decision)

# TODO

class GANGenerator(Tasklet):
  def __init__(self, generator, discriminator, run=None, loss=None):
    super().__init__()
    self.gen = Generator(generator)
    self.disc = Discriminator(
      discriminator,
      run=run or DiscriminatorRun(real=False),
      loss=loss or DiscriminatorLoss(real=False),
      real=False
    )

  def run(self, sample):
    make_differentiable(sample.noise)
    fake = self.gen.run(sample.noise, sample.args)
    decision = self.disc.run(fake)
    return NamedTuple(
      fake=fake,
      sample=sample,
      decision=decision
    )

  def loss(self, decision):
    loss = self.disc.loss(*decision)
    self.store(generator_loss=loss)
    return loss

  def step(self, sample):
    result = self.run(sample)
    loss = self.loss(result.decision)
    return loss

class GANDiscriminator(Tasklet):
  def __init__(self, discriminator, run=None, loss=None):
    super().__init__()
    self.disc = Discriminator(
      discriminator,
      run=run or DiscriminatorRun(real=True),
      loss=loss or DiscriminatorLoss(real=True),
      real=False
    )

  def realness(self, decision, real=True):
    return int(real) - torch.ones_like(decision)

  def run(self, real, fake):
    make_differentiable(real.data)
    make_differentiable(fake.data)
    decision = self.disc.run(fake, real)
    return decision

  def loss(self, fake_decision, real_decision):
    loss = self.disc.loss(fake_decision, real_decision)
    self.store(discriminator_total_loss=loss)
    return loss

  def step(self, fake, real):
    result = self.run(fake, real)
    loss = self.loss(*result)
    return loss
