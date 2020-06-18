from torchsupport.interacting.off_energy_training import OffEnergyTraining

class OffEBMTraining(OffEnergyTraining):
  def run_score(self, sample, real):
    fake = sample.final_state
    fake_sample_energy = sample.final_energy
    fake_energy = self.score(fake)
    real_energy = self.score(real)
    energy_difference = fake_energy - fake_sample_energy

    return real_energy, fake_energy, energy_difference

  def score_loss(self, real_energy, fake_energy, energy_difference):
    regularization = self.decay * ((real_energy ** 2).mean() + (fake_energy ** 2).mean())

    fake_weight = torch.exp(0.1 * energy_difference).detach()
    weight_sum = fake_weight.sum()

    real_mean = real_energy.mean()
    fake_mean = (fake_energy * fake_weight).sum() / weight_sum

    ebm = real_mean - fake_mean
    self.current_losses["real"] = float(real_mean)
    self.current_losses["fake"] = float(fake_mean)
    self.current_losses["fake raw"] = float(fake_energy.mean())
    self.current_losses["regularization"] = float(regularization)
    self.current_losses["ebm"] = float(ebm)
    return regularization + ebm

  def run_auxiliary(self, data):
    pass

  def auxiliary_loss(self, *args):
    return 0.0
