
import numpy as np
import torch
from torch import nn
from torch.nn import functional as func
from torch.distributions import Normal, RelaxedOneHotCategorical

from tensorboardX import SummaryWriter

from torchsupport.training.training import Training
import torchsupport.modules.losses.vae as vl
from torchsupport.data.io import netwrite, make_differentiable
from torchsupport.data.collate import DataLoader

from torchsupport.training.neural_conditioner import NeuralConditionerTraining

class ConsistentGANTraining(NeuralConditionerTraining):
  def __init__(self, generator, discriminator, data, levels=None, gamma=1, **kwargs):
    super().__init__(generator, discriminator, data, **kwargs)
    self.gamma = gamma
    self.levels = levels

  def sample(self, data):
    noise = super().sample(data)
    return noise

  def zoom(self, data, idx):
    shape = data.shape[-1]
    pos = (shape - self.levels[idx]) // 2
    return data[:, :, pos:pos + self.levels[idx], pos:pos + self.levels[idx]]

  def stage(self, data):
    inputs = data
    stages = [
      self.zoom(layer, idx)
      for idx, layer in enumerate(inputs[:-1])
    ]
    return stages

  def restrict_inputs(self, inputs, mask):
    return [
      inp * msk
      for inp, msk in zip(inputs, mask)
    ]

  def reconstruction_loss(self, generated, stages):
    l1_loss = 0.0
    for idx, stage in enumerate(stages):
      compare = generated[idx + 1]
      compare = func.adaptive_avg_pool2d(compare, stage.shape[-1])
      diff = (compare - stage).view(compare.size(0), -1).norm(p=1, dim=1)
      l1_loss += diff.mean()
    self.current_losses["reconstruction"] = float(l1_loss)
    return l1_loss

  def run_generator(self, data):
    sample = self.sample(data)
    inputs, available, requested = data
    stages = self.stage(inputs)
    restricted_inputs = self.restrict_inputs(inputs, available)
    generated, stages = self.generator(
      sample, stages, restricted_inputs,
      available, requested
    )

    return inputs, generated, stages, available, requested

  def run_discriminator(self, data):
    with torch.no_grad():
      fake = self.run_generator(data)
    make_differentiable(fake)
    make_differentiable(data)
    _, fake_batch, _, _, _ = fake
    inputs, available, requested = data
    fake_result = self._run_discriminator_aux(
      inputs, fake_batch,
      available, requested
    )
    real_result = self._run_discriminator_aux(
      inputs, inputs,
      available, requested
    )
    return fake, data, fake_result, real_result

  def generator_step_loss(self, data, generated, stages, avl, req):
    gan_loss = super().generator_step_loss(data, generated, avl, req)
    reconstruction_loss = self.reconstruction_loss(generated, stages)
    return gan_loss + self.gamma * reconstruction_loss
