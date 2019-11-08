import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as func
from torch.distributions import Normal, RelaxedOneHotCategorical

from torchsupport.training.state import (
  NetState, NetNameListState, TrainingState
)
from torchsupport.training.gan import RothGANTraining
import torchsupport.modules.losses.vae as vl
from torchsupport.data.io import netwrite, to_device, detach, make_differentiable
from torchsupport.data.collate import DataLoader

class FewShotGANTraining(RothGANTraining):
  def mixing_key(self, data):
    return data[1]

  def sample(self, data):
    the_generator = self.generator
    if isinstance(the_generator, nn.DataParallel):
      the_generator = the_generator.module
    return to_device(the_generator.sample(data), self.device)

  def divergence_loss(self, sample):
    _, encoder_parameters = sample
    result = vl.normal_kl_norm_loss(*encoder_parameters)
    return result

  def generator_step_loss(self, data, generated, sample):
    gan_loss = super().generator_step_loss(data, generated, sample)
    sample_divergence_loss = self.divergence_loss(sample)
    self.current_losses["kullback leibler"] = float(sample_divergence_loss)
    return gan_loss + sample_divergence_loss
