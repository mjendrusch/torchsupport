
import numpy as np
import torch
from torch import nn
from torch.nn import functional as func
from torch.distributions import Normal, RelaxedOneHotCategorical

from tensorboardX import SummaryWriter

from torchsupport.training.training import Training
import torchsupport.modules.losses.vae as vl
from torchsupport.data.io import netwrite
from torchsupport.data.collate import DataLoader

from torchsupport.training.gan import GANTraining

class NeuralConditionerTraining(GANTraining):
  def __init__(self, generator, discriminator, data, **kwargs):
    super(NeuralConditionerTraining, self).__init__(
      generator,
      discriminator,
      data, **kwargs
    )

  def discriminator_loss(self, data, fake, fake_res, real_res):
    loss = GANTraining.discriminator_loss(self, data, fake, fake_res, real_res)
    regularizer = 0.5 * (fake_res.sigmoid().mean(dim=0) + real_res.sigmoid().mean(dim=0))
    loss += regularizer.mean(dim=0)
    return loss, None

  def generator_loss(self, inputs, generated, available, requested):
    discriminator_result = self._run_discriminator_aux(
      inputs, generated, available, requested
    )
    loss_val = func.binary_cross_entropy_with_logits(
      discriminator_result,
      torch.ones(self.batch_size, 1).to(self.device)
    )

    return loss_val

  def restrict_inputs(self, data, mask):
    return data * mask

  def run_generator(self, data):
    sample = self.sample()
    inputs, available, requested = data
    restricted_inputs = self.restrict_inputs(inputs, available)
    generated = self.generator(
      sample, restricted_inputs,
      available, requested
    )

    return inputs, generated, available, requested

  def _run_discriminator_aux(self, x, x_p, a, r):
    avail = self.restrict_inputs(x, a)
    reqst = self.restrict_inputs(x_p, r)
    result = self.discriminator(avail, reqst, a, r)
    return result

  def run_discriminator(self, data):
    _, fake_batch, _, _ = self.run_generator(data)
    inputs, available, requested = data
    fake_result = self._run_discriminator_aux(
      inputs, fake_batch,
      available, requested
    )
    real_result = self._run_discriminator_aux(
      inputs, inputs,
      available, requested
    )
    return data, fake_batch, fake_result, real_result
