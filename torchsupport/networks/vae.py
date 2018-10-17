import torch
import torch.nn as nn
import torch.nn.functional as func
from torchsupport.ops.shape import flatten

class VAELoss(nn.Module):
  def __init__(self, reconstruction_loss):
    super(VAELoss, self).__init__()
    self.reconstruction_loss = reconstruction_loss
  
  def forward(self, reconstruction, target, mu, logvar):
    reconstruction_loss = self.reconstruction_loss(reconstruction, target)
    relative_entropy = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + relative_entropy

class VAE(nn.Module):
  def __init__(self, encoder, decoder, latent_space):
    super(VAE, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.latent_space = latent_space
    output_size = self.encoder.output_size
    if output_size == None:
      self.encoder = lambda x: flatten(func.adaptive_avg_pool2d(self.encoder(x), (1, 1)))
      output_size = self.encoder.output_layers
    self.reparameterizer_sigma = nn.LinearLayer(output_size, latent_space)
    self.reparameterizer_mu = nn.LinearLayer(output_size, latent_space)

  def reparameterize(self, mu, sigma):
    sigma = sigma.mul(0.5).exp_() 
    epsilon = torch.FloatTensor(sigma.size()).normal_().to(latent_vector.device())
    return epsilon.mul(sigma).add_(mu)

  def encode(self, input):
    return self.encoder(input)

  def change_parameters(self, latent_vector):
    logvar = self.reparameterizer_sigma(latent_vector)
    mu = self.reparameterizer_mu(latent_vector)
    return mu, logvar

  def encode_inner(self, input):
    latent_vector = self.encode(input)
    mu, logvar = self.change_parameters(latent_vector)
    return mu, logvar

  def decode_inner(self, mu, sigma):
    reparameterization = self.reparameterize(mu, sigma)
    return self.decoder(reparameterization)

  def decode(self, latent_vector):
    mu, logvar = self.change_parameters(latent_vector)
    return self.decode_inner(mu, logvar)

  def forward(self, input):
    mu, logvar = self.encode_inner(input)
    reconstruction = self.decode_inner(mu, logvar)
    return reconstruction, mu, logvar
