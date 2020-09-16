
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

from torchsupport.training.gan import GANTraining, RothGANTraining

def _gradient_norm(inputs, parameters):
  out = torch.ones(inputs.size()).to(inputs.device)
  gradients = torch.autograd.grad(
    inputs, parameters, create_graph=True, retain_graph=True,
    grad_outputs=out
  )
  grad_sum = 0.0
  for gradient in gradients:
    grad_sum += (gradient ** 2).view(gradient.size(0), -1).sum(dim=1)
  grad_sum = torch.sqrt(grad_sum + 1e-16)
  return grad_sum, out

class NeuralConditionerTraining(RothGANTraining):
  def __init__(self, generator, discriminator, data, **kwargs):
    super(NeuralConditionerTraining, self).__init__(
      generator,
      discriminator,
      data, **kwargs
    )

  def mixing_key(self, data):
    if len(data) == 4:
      return data[1]
    else:
      return data[0]

  def regularization(self, fake, real, generated_result, real_result):
    real_norm, real_out = _gradient_norm(real_result, self.mixing_key(real))
    fake_norm, fake_out = _gradient_norm(generated_result, self.mixing_key(fake))

    real_penalty = real_norm ** 2
    fake_penalty = fake_norm ** 2

    penalty = 0.5 * (real_penalty + fake_penalty).mean()

    out = (real_out, fake_out)

    return penalty, out

  def generator_loss(self, inputs, generated, available, requested):
    discriminator_result = self._run_discriminator_aux(
      inputs, generated, available, requested
    )
    loss_val = func.binary_cross_entropy_with_logits(
      discriminator_result,
      torch.zeros_like(discriminator_result).to(self.device)
    )

    return loss_val

  def restrict_inputs(self, data, mask):
    return data * mask

  def run_generator(self, data):
    sample = self.sample(data)
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
    with torch.no_grad():
      fake = self.run_generator(data)
    make_differentiable(fake)
    make_differentiable(data)
    _, fake_batch, _, _ = fake
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
