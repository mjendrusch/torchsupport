import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as func
from torch.utils.data import DataLoader
from torch.distributions import Normal, RelaxedOneHotCategorical

from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

from torchsupport.training.vae import VAETraining
import torchsupport.modules.losses.vae as vl
from torchsupport.data.io import netwrite

class NPTraining(VAETraining):
  def __init__(self, encoder, decoder, aggregator, data,
               rec_loss=func.binary_cross_entropy_with_logits, **kwargs):
    super(NPTraining, self).__init__(encoder, decoder, data, **kwargs)
    self.aggregator = aggregator
    self.rec_loss = rec_loss

  def loss(self, source_parameters, total_parameters, reconstruction, target):
    loss_val = self.rec_loss(reconstruction, target)
    kld = vl.normal_kl_loss(total_parameters, source_parameters)
    return loss_val + kld

  def run_networks(self, data):
    xs, ys, source_indices, target_indices = data
    representation = self.encoder(xs, ys)
    s_mean, s_logvar = self.aggregator(representation, source_indices)
    t_mean, t_logvar = self.aggregator(representation, source_indices + target_indices)
    target_access = target_indices.nonzero()
    target = ys[target_access]
    reconstruction = self.decoder(xs[target_indices.nonzero()])
    return (s_mean, s_logvar), (t_mean, t_logvar), reconstruction, target
