import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.data.io import to_device, make_differentiable
from torchsupport.data.collate import DataLoader, default_collate
from torchsupport.training.energy import AbstractEnergyTraining

class DenoisingDiffusionIntegrator:
  def __init__(self, step_factor=1):
    self.noise_weights = ...
    self.step_factor = step_factor

  def integrate(self, score, data, *args):
    with torch.no_grad():
      noise_weights = [
        self.noise_weights[idx]
        for idx in reversed(range(0, len(self.noise_weights), self.step_factor))
      ]
      for idx, (a_0, a_1) in enumerate(
          zip(noise_weights, noise_weights[1:] + [torch.ones(1)[0]])
      ):
        # time = idx * torch.ones(data.size(0)).to(data.device)
        # correction = score(data, time, *args)
        # denoised = (data - correction * (1 - a_0).sqrt()) / a_0.sqrt()
        # data = a_1.sqrt() * denoised + (1 - a_1).sqrt() * correction
        # print(correction.min(), correction.max(), correction.mean(), correction.std())
        # print(denoised.min(), denoised.max(), denoised.mean(), denoised.std())
        # print(data.min(), data.max(), data.mean(), data.std())
        step = a_1.sqrt() * ((((1 - a_1) / a_1).sqrt() - ((1 - a_0) / a_0).sqrt()))
        scale = (a_1 / a_0).sqrt()
        time = idx * torch.ones(data.size(0)).to(data.device)
        data = scale * data + step * score(data, time, *args)
      return data

class DenoisingDiffusionTraining(AbstractEnergyTraining):
  def __init__(self, score, *args,
               integrator=None,
               noise_weights=None,
               timesteps=1000,
               skipsteps=1,
               optimizer=torch.optim.Adam,
               **kwargs):
    self.score = ...
    super().__init__(
      {"score": score}, *args, optimizer=optimizer, **kwargs
    )
    self.integrator = integrator or DenoisingDiffusionIntegrator(step_factor=skipsteps)
    self.noise_weights = noise_weights
    self.skipsteps = skipsteps
    if self.noise_weights is None:
      self.noise_weights = self.beta_noise_weights(steps=timesteps)
    self.integrator.noise_weights = self.noise_weights

  @staticmethod
  def beta_noise_weights(start=0.02, end=1e-4, steps=1000):
    betas = torch.tensor(np.linspace(end, start, steps, dtype=np.float32))
    betas = torch.cat((torch.zeros(1), betas), dim=0)
    alphas = (1 - betas).cumprod(dim=0)
    return alphas[1:]

  def step(self, data):
    data = to_device(data, self.device)
    data, *args = self.data_key(data)
    self.energy_step(data, args)
    self.each_step()

  def run_energy(self, data, args):
    times = torch.randint(0, len(self.noise_weights) // self.skipsteps, (self.batch_size // 2 + 1,))
    times = torch.cat((times, len(self.noise_weights) // self.skipsteps - times - 1), dim=0)[:self.batch_size].to(self.device)
    times = self.skipsteps * times
    expansion = [slice(None)] + [None] * (data.dim() - 1)
    weights = self.noise_weights[times][expansion].to(self.device)
    noise = torch.randn_like(data).to(self.device)
    noised = weights.sqrt() * data + (1 - weights).sqrt() * noise
    result = self.score(noised, times, *args)
    return result, noise

  def energy_loss(self, prediction, noise):
    result = ((prediction - noise) ** 2).view(prediction.size(0), -1).mean(dim=0).sum()
    self.current_losses["denoising"] = float(result)
    return result

  def energy_step(self, data, args):
    self.optimizer.zero_grad()
    prediction, noise = self.run_energy(data, args)
    loss = self.energy_loss(prediction, noise)
    self.log_statistics(float(loss))
    loss.backward()
    self.optimizer.step()

  def sample(self):
    self.score.eval()
    batch = next(iter(self.train_data))
    data, *args = self.data_key(to_device(batch, self.device))
    data = torch.randn_like(data)

    improved = self.integrator.integrate(self.score, data, *args).detach()
    self.score.train()
    return (improved, *args)

  def each_step(self):
    super().each_step()
    if self.step_id % self.report_interval == 0 and self.step_id != 0:
      data, *args = self.sample()
      self.each_generate(data, *args)

  def each_generate(self, improved, *args):
    shift = ((improved / improved.std()).clamp(-1, 1) + 1) / 2
    if shift.size(1) == 1:
      shift = shift.repeat_interleave(3, dim=1)
    self.writer.add_images("shift", shift, self.step_id)
    shift = ((improved).clamp(-1, 1) + 1) / 2
    if shift.size(1) == 1:
      shift = shift.repeat_interleave(3, dim=1)
    self.writer.add_images("basic", shift, self.step_id)
   
  def data_key(self, data):
    result, *args = data
    return (result, *args)
