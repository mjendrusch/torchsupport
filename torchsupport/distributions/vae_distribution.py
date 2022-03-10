import torch
import torch.nn as nn
from torch.distributions import Distribution

class VAEDistribution(nn.Module, Distribution):
  def __init__(self, encoder, decoder, prior=None):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.prior = prior

  def forward(self, inputs):
    q = self.encoder(inputs)
    sample = q.rsample()
    d = self.decoder(sample)
    rec = d.log_prob(inputs).view(inputs.size(0), -1).sum(dim=1, keepdim=True)
    log_p = self.prior.log_prob(sample).view(inputs.size(0), -1)
    log_q = q.log_prob(sample).view(inputs.size(0), -1)
    kl = -log_p.sum(dim=1, keepdim=True) + log_q.sum(dim=1, keepdim=True)
    return rec - kl

  def log_prob(self, x):
    return self(x)

  def rsample(self, sample_shape=torch.Size()):
    prior = self.prior.rsample(sample_shape=sample_shape)
    decoded = self.decoder(prior)
    return decoded.rsample()

  def sample(self, sample_shape=torch.Size()):
    with torch.no_grad():
      return self.rsample(sample_shape=sample_shape)
