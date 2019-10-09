import numpy as np
import torch
from torch import nn
from torch.nn import functional as func
from torch.distributions import Normal, RelaxedOneHotCategorical

from tensorboardX import SummaryWriter
import torchsupport.modules.losses.vae as vl
import torchsupport.ops.distance as kernels
from torchsupport.training.vae import VAETraining

class TransformingVAETraining(VAETraining):
  """
  Attempt at adapting http://arxiv.org/abs/1910.01791
  """

  def __init__(self, encoder, decoder, data, domain_scale=1, **kwargs):
    self.domain_scale = domain_scale
    super(TransformingVAETraining, self).__init__(encoder, decoder, data, **kwargs)

  def kernel(self, x, y):
    return kernels.multi_rbf_distance_matrix(x, y) 

  def preprocess(self, data):
    data, condition, domain = data
    return data, condition, domain

  def domain_divergence_loss(self, cond_latent, domain):
    """
    Gross simplification of their MMD loss by adding the losses on the distance matrix level 
    """

    eq = domain[None, :] == domain[:, None] 
    eq = eq.to(torch.float) * 2 - 1
    mmd = self.kernel(cond_latent, cond_latent) * eq
    return mmd.mean()

  def loss(self, mean, logvar, reconstruction, target, domain):
    rec, cond_latent = reconstruction
    ce = self.reconstruction_loss(rec, target)
    kld = self.divergence_loss(mean, logvar)
    ddl = self.domain_divergence_loss(cond_latent, domain)
    loss_val = ce + kld + self.domain_scale * ddl
    self.current_losses['cross-entropy'] = float(ce)
    self.current_losses['kullback-leibler'] = float(kld)
    self.current_losses['domain-divergence'] = float(ddl)
    return loss_val

  def run_networks(self, target, condition, domain):
    res = super(TransformingVAETraining, self).run_networks(target, condition, domain)
    return (*res, domain)
