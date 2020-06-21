import torch

from torchsupport.interacting.off_energy_training import OffEnergyTraining

class OffEBMTraining(OffEnergyTraining):
  def __init__(self, score, energy, data,
               off_energy_weight=0.0,
               off_energy_decay=0.1,
               auxiliary_steps=0, **kwargs):
    super().__init__(
      score, energy, data,
      auxiliary_steps=auxiliary_steps,
      **kwargs
    )
    self.off_energy_weight = off_energy_weight
    self.off_energy_decay = off_energy_decay

  def run_score(self, sample, data):
    fake = sample.final_state
    fake_args = sample.args or []
    real, *real_args = data
    fake_sample_energy = sample.final_energy
    fake_energy = self.score(fake, *fake_args)
    real_energy = self.score(real, *real_args)
    energy_difference = fake_energy - fake_sample_energy

    return real_energy, fake_energy, energy_difference

  def score_loss(self, real_energy, fake_energy, energy_difference):
    regularization = self.decay * ((real_energy ** 2).mean() + (fake_energy ** 2).mean())

    fake_weight = torch.exp(
      -self.off_energy_weight * abs(energy_difference)
    ).detach()
    weight_sum = fake_weight.sum()
    weight_mean = weight_sum / fake_weight.size(0)

    real_mean = real_energy.mean()
    fake_mean = (fake_energy * fake_weight).sum() / weight_sum

    off_energy_loss = self.off_energy_decay * (energy_difference ** 2).mean()

    ebm = real_mean - fake_mean
    self.current_losses["real"] = float(real_mean)
    self.current_losses["weight"] = float(weight_mean)
    self.current_losses["off energy"] = float(off_energy_loss)
    self.current_losses["energy difference"] = float(abs(energy_difference).mean())
    self.current_losses["fake"] = float(fake_mean)
    self.current_losses["fake raw"] = float(fake_energy.mean())
    self.current_losses["regularization"] = float(regularization)
    self.current_losses["ebm"] = float(ebm)
    return regularization + ebm + off_energy_loss

  def run_auxiliary(self, data):
    pass

  def auxiliary_loss(self, *args):
    return 0.0
