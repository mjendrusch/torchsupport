import torch

from torchsupport.data.io import make_differentiable
from torchsupport.training.gan import (
  RothGANTraining, AbstractGANTraining, GANTraining
)

class PairedGANTraining(RothGANTraining):
  def __init__(self, generator, discriminator, data, gamma=100, **kwargs):
    super().__init__(generator, discriminator, data, **kwargs)
    self.gamma = gamma

  def mixing_key(self, data):
    return data[1]

  def sample(self, data):
    noise = super().sample(data)
    return noise, data[0]

  def reconstruction_loss(self, data, generated, sample):
    diff = abs(data[1] - generated[1]).view(data[1].size(0), -1)
    l1_loss = diff.mean()
    return l1_loss

  def generator_step_loss(self, data, generated, sample):
    gan_loss = super().generator_step_loss(data, generated, sample)
    reconstruction_loss = self.reconstruction_loss(data, generated, sample)
    return gan_loss + self.gamma * reconstruction_loss

class CycleGANTraining(RothGANTraining):
  def __init__(self, generators, discriminators, data, gamma=10, **kwargs):
    self.fw = ...
    self.rv = ...
    self.fw_discriminator = ...
    self.rv_discriminator = ...
    self.discriminator = ...
    AbstractGANTraining.__init__(
      self,
      {"fw": generators[0], "rv": generators[1]},
      {"fw_discriminator": discriminators[0], "rv_discriminator": discriminators[1]},
      data, **kwargs
    )
    self.generator = self.fw

    self.gamma = gamma

  def set_discriminator(self, disc):
    self.discriminator = disc

  def cycle_loss(self, data, cycled):
    l1 = (data - cycled).view(data.size(0), -1).norm(p=1, dim=1)
    return l1.mean()

  def generator_step_loss(self, data, translated, cycled):
    self.set_discriminator(self.fw_discriminator)
    loss_fw = self.generator_loss(data[0], translated[0])
    loss_cycle_fw = self.cycle_loss(data[0], cycled[0])

    self.set_discriminator(self.rv_discriminator)
    loss_rv = self.generator_loss(data[1], translated[1])
    loss_cycle_rv = self.cycle_loss(data[1], cycled[1])

    loss_gan = loss_fw + loss_rv
    loss_cycle = loss_cycle_fw + loss_cycle_rv

    self.current_losses["cycle"] = float(loss_cycle)
    self.current_losses["gan"] = float(loss_gan)

    return loss_gan + self.gamma * loss_cycle

  def discriminator_step_loss(self, translated, data, translated_result, real_result):
    self.set_discriminator(self.fw_discriminator)
    loss_fw, out_fw = self.discriminator_loss(
      translated[0], data[0], translated_result[0], real_result[0]
    )
    self.current_losses["fw_discriminator"] = float(loss_fw)

    self.set_discriminator(self.rv_discriminator)
    loss_rv, out_rv = self.discriminator_loss(
      translated[1], data[1], translated_result[1], real_result[1]
    )
    self.current_losses["rv_discriminator"] = float(loss_rv)

    loss = loss_fw + loss_rv
    out = (out_fw, out_rv)

    self.current_losses["discriminator"] = float(loss)

    return loss, out

  def run_generator(self, data):
    translated_fw = self.fw(self.sample(data), data[0])
    cycled_fw = self.rv(self.sample(data), translated_fw)

    translated_rv = self.rv(self.sample(data), data[1])
    cycled_rv = self.fw(self.sample(data), translated_rv)

    translated = (translated_fw, translated_rv)
    cycled = (cycled_fw, cycled_rv)

    return data, translated, cycled

  def run_discriminator(self, data):
    with torch.no_grad():
      _, (fake_fw, fake_rv), _ = self.run_generator(data)
    make_differentiable(fake_fw)
    make_differentiable(fake_rv)
    make_differentiable(data)
    real_result_fw = self.fw_discriminator(data[1])
    fake_result_fw = self.fw_discriminator(fake_fw)
    real_result_rv = self.rv_discriminator(data[0])
    fake_result_rv = self.rv_discriminator(fake_rv)

    real_result = (real_result_fw, real_result_rv)
    fake_result = (fake_result_fw, fake_result_rv)
    fake_batch = fake_fw, fake_rv
    real_batch = (data[1], data[0])

    return fake_batch, real_batch, fake_result, real_result

class AugmentedCycleGANTraining(CycleGANTraining):
  def __init__(self, generators, discriminators, encoders, data, gamma=10, **kwargs):
    self.fw = ...
    self.rv = ...
    self.fw_encoder = ...
    self.rv_encoder = ...
    self.fw_discriminator = ...
    self.rv_discriminator = ...
    self.z_fw_discriminator = ...
    self.z_rv_discriminator = ...
    self.discriminator = ...
    AbstractGANTraining.__init__(
      self,
      {
        "fw": generators[0],
        "rv": generators[1],
        "fw_encoder": encoders[0],
        "rv_encoder": encoders[1]
      },
      {
        "fw_discriminator": discriminators[0],
        "rv_discriminator": discriminators[1],
        "z_fw_discriminator": discriminators[2],
        "z_rv_discriminator": discriminators[3]
      },
      data, **kwargs
    )
    self.generator = self.fw

    self.gamma = gamma

  def run_cycle(self, data, fw, rv, fw_enc, rv_enc):
    z_fw = fw_enc.sample(self.batch_size)
    translated_fw = fw(z_fw, data)
    z_rv = rv_enc(data, translated_fw)
    cycled_fw = rv(z_rv, translated_fw)
    z_cycled_fw = fw_enc(data, translated_fw)

    return data, translated_fw, cycled_fw, z_fw, z_rv, z_cycled_fw

  def generator_step_loss(self, data, translated, cycled, z_fw, z_rv, z_cycled):
    self.set_discriminator(self.fw_discriminator)
    loss_fw = self.generator_loss(data[0], translated[0])
    loss_cycle_fw = self.cycle_loss(data[0], cycled[0])

    self.set_discriminator(self.z_fw_discriminator)
    loss_z_fw = self.generator_loss(z_fw[0], z_rv[0])
    loss_z_cycle_fw = self.cycle_loss(z_fw[0], z_cycled[0])

    self.set_discriminator(self.rv_discriminator)
    loss_rv = self.generator_loss(data[1], translated[1])
    loss_cycle_rv = self.cycle_loss(data[1], cycled[1])

    self.set_discriminator(self.z_rv_discriminator)
    loss_z_rv = self.generator_loss(z_fw[1], z_rv[1])
    loss_z_cycle_rv = self.cycle_loss(z_fw[1], z_cycled[1])

    loss_gan = loss_fw + loss_rv + loss_z_fw + loss_z_rv
    loss_cycle = loss_cycle_fw + loss_cycle_rv + loss_z_cycle_fw + loss_z_cycle_rv

    self.current_losses["cycle"] = float(loss_cycle)
    self.current_losses["gan"] = float(loss_gan)

    return loss_gan + self.gamma * loss_cycle

  def discriminator_step_loss(self, translated, data, translated_result, real_result):
    self.set_discriminator(self.fw_discriminator)
    loss_fw, out_fw = self.discriminator_loss(
      translated[0], data[0], translated_result[0], real_result[0]
    )
    self.current_losses["fw_discriminator"] = float(loss_fw)

    self.set_discriminator(self.rv_discriminator)
    loss_rv, out_rv = self.discriminator_loss(
      translated[1], data[1], translated_result[1], real_result[1]
    )
    self.current_losses["rv_discriminator"] = float(loss_rv)

    self.set_discriminator(self.z_fw_discriminator)
    loss_z_fw, out_z_fw = self.discriminator_loss(
      translated[2], data[2], translated_result[2], real_result[2]
    )
    self.current_losses["z_fw_discriminator"] = float(loss_fw)

    self.set_discriminator(self.z_rv_discriminator)
    loss_z_rv, out_z_rv = self.discriminator_loss(
      translated[3], data[3], translated_result[3], real_result[3]
    )
    self.current_losses["z_rv_discriminator"] = float(loss_rv)

    loss = loss_fw + loss_rv + loss_z_fw + loss_z_rv
    out = (out_fw, out_rv, out_z_fw, out_z_rv)

    self.current_losses["discriminator"] = float(loss)

    return loss, out

  def run_generator(self, data):
    args_fw = self.run_cycle(
      data[0], self.fw, self.rv, self.fw_encoder, self.rv_encoder
    )
    args_rv = self.run_cycle(
      data[1], self.rv, self.fw, self.rv_encoder, self.fw_encoder
    )
    return tuple(zip(args_fw, args_rv))    

  def run_discriminator(self, data):
    with torch.no_grad():
      _, (fake_fw, fake_rv), _, (real_z_fw, real_z_rv), (fake_z_fw, fake_z_rv), _  = \
        self.run_generator(data)
    make_differentiable(fake_fw)
    make_differentiable(fake_rv)
    make_differentiable(real_z_fw)
    make_differentiable(real_z_rv)
    make_differentiable(fake_z_fw)
    make_differentiable(fake_z_rv)
    make_differentiable(data)
    real_result_fw = self.fw_discriminator(data[1])
    fake_result_fw = self.fw_discriminator(fake_fw)
    real_result_rv = self.rv_discriminator(data[0])
    fake_result_rv = self.rv_discriminator(fake_rv)
    real_result_z_fw = self.z_fw_discriminator(real_z_fw)
    fake_result_z_fw = self.z_fw_discriminator(fake_z_fw)
    real_result_z_rv = self.z_rv_discriminator(real_z_rv)
    fake_result_z_rv = self.z_rv_discriminator(fake_z_rv)

    real_result = (real_result_fw, real_result_rv, real_result_z_fw, real_result_z_rv)
    fake_result = (fake_result_fw, fake_result_rv, fake_result_z_fw, fake_result_z_rv)
    fake_batch = fake_fw, fake_rv, fake_z_fw, fake_z_rv
    real_batch = (data[1], data[0], real_z_fw, real_z_rv)

    return fake_batch, real_batch, fake_result, real_result
