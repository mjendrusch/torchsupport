import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal

from torchvision.models import vgg16

from torchsupport.data.io import make_differentiable, to_device
from torchsupport.data.match import match
from torchsupport.training.gan import (
  RothGANTraining, AbstractGANTraining, GANTraining
)

class VQGANGenerator(nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.store = None

  @property
  def loss(self):
    result = self.store
    self.store = None
    return result

  @loss.setter
  def loss(self, value):
    self.store = value

  @property
  def last_weight(self):
    return self.decoder.last_weight

  def forward(self, inputs):
    code, index, assignment_cost = self.encoder(inputs)
    self.loss = assignment_cost
    reconstruction = self.decoder(code)
    return reconstruction

class VAEGANGenerator(nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.store = None

  @property
  def loss(self):
    result = self.store
    self.store = None
    return result

  @loss.setter
  def loss(self, value):
    self.store = value

  @property
  def last_weight(self):
    return self.decoder.last_weight

  def forward(self, inputs):
    posterior = self.encoder(inputs)
    code = posterior.rsample()
    prior = Normal(torch.zeros_like(code), torch.ones_like(code))
    self.loss = match(posterior, prior)
    reconstruction = self.decoder(code)
    return reconstruction

class VGGPerceptual(nn.Module):
  def __init__(self):
    super().__init__()
    vgg_pretrained_features = vgg16(pretrained=True).features.eval()
    self.slices = []
    self.slices.append(nn.Sequential(*vgg_pretrained_features[:4]))
    self.slices.append(nn.Sequential(*vgg_pretrained_features[4:9]))
    self.slices.append(nn.Sequential(*vgg_pretrained_features[9:16]))
    self.slices.append(nn.Sequential(*vgg_pretrained_features[16:23]))
    self.slices.append(nn.Sequential(*vgg_pretrained_features[23:30]))
    self.slices = nn.ModuleList(self.slices)
    for param in self.parameters():
      param.requires_grad = False
    self.logv = nn.Parameter(torch.zeros(6, requires_grad=True))

  def forward(self, x, y):
    loss = 0.0
    ox, oy = x, y
    dist = Normal(ox, self.logv[0].exp())
    loss += dist.log_prob(oy).mean(dim=0).mean()
    for slc, logv in zip(self.slices, self.logv[1:]):
      ox, oy = slc(ox), slc(oy)
      ox = ox / (ox.norm(dim=1, keepdim=True) + 1e-6)
      oy = oy / (oy.norm(dim=1, keepdim=True) + 1e-6)
      dist = Normal(ox, self.logv[0].exp())
      loss += dist.log_prob(oy).mean(dim=0).mean()
    return -loss / 6

class VQGANTraining(RothGANTraining):
  def __init__(self, encoder, decoder, discriminator, data,
               perceptual=None, code_weight=1.0, generator=None,
               **kwargs):
    generator = generator or VQGANGenerator
    super().__init__(
      generator(encoder, decoder),
      discriminator, data, **kwargs
    )
    self.code_weight = code_weight
    self.perceptual = (perceptual or VGGPerceptual()).to(self.device)
    pgroup = {"params": self.perceptual.parameters()}
    self.generator_optimizer.add_param_group(pgroup)

  def mixing_key(self, data):
    return data[0]

  def sample(self, data):
    return (data[0],)

  def weight(self, gan, rec):
    gan_weight = torch.autograd.grad(gan, self.generator.last_weight, retain_graph=True)[0]
    gan_weight = gan_weight.norm()
    rec_weight = torch.autograd.grad(rec, self.generator.last_weight, retain_graph=True)[0]
    rec_weight = rec_weight.norm()
    balance = (rec_weight / (gan_weight + 1e-4)).clamp_max(1e4)
    return balance

  def reconstruction_loss(self, data, generated, sample):
    loss = self.perceptual(data[0], generated[0])
    return loss

  def generator_step_loss(self, data, generated, sample):
    gan_loss = super().generator_step_loss(data, generated, sample).mean()
    rec_loss = self.reconstruction_loss(data, generated, sample).mean()
    code_loss = self.generator.loss.mean()
    self.current_losses["gan"] = float(gan_loss)
    self.current_losses["rec"] = float(rec_loss)
    self.current_losses["code"] = float(code_loss)
    weight = self.weight(gan_loss, rec_loss)
    self.current_losses["weight"] = float(weight)
    return rec_loss + self.code_weight * code_loss + weight * gan_loss
