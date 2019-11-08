import random

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset
from torch.distributions import Normal

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from torchsupport.modules.basic import MLP
from torchsupport.modules.residual import ResNetBlock2d
from torchsupport.training.samplers import Langevin
from torchsupport.training.few_shot_gan import FewShotGANTraining

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

class EnergyDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __getitem__(self, index):
    data, label_index = self.data[index]
    label = torch.zeros(10)
    label[label_index] = 1
    return data, label

  def __len__(self):
    return len(self.data)

class MNISTSet(EnergyDataset):
  def __init__(self, data, size=5):
    super().__init__(data)
    self.size = size

  def __getitem__(self, index):
    data = []
    label = random.randrange(10)
    for idx in range(self.size):
      d, l = super().__getitem__(random.randrange(len(self)))
      while l[label] < 1.0:
        d, l = super().__getitem__(random.randrange(len(self)))
      data.append(d.unsqueeze(0))
    data = torch.cat(data, dim=0)
    return data, data

class SingleEncoder(nn.Module):
  def __init__(self, latents=32):
    super(SingleEncoder, self).__init__()
    self.block = MLP(28 * 28, latents, hidden_size=64, depth=4)

  def forward(self, inputs):
    return self.block(inputs)

class Encoder(nn.Module):
  def __init__(self, single, size=5, latents=16):
    super(Encoder, self).__init__()
    self.size = size
    self.single = single
    self.weight = nn.Linear(32, 1)
    self.combine = MLP(32, 32, 64, depth=3)
    self.mean = nn.Linear(32, latents)
    self.logvar = nn.Linear(32, latents)

  def forward(self, inputs):
    inputs = inputs.view(-1, 28 * 28)
    out = self.single(inputs)
    weights = self.weight(out)
    out = out.view(-1, self.size, 32)
    weights = weights.view(-1, self.size, 1).softmax(dim=1)
    pool = (weights * out).sum(dim=1)
    pool = self.combine(pool)
    return self.mean(pool), self.logvar(pool)

class Generator(nn.Module):
  def __init__(self, size=5):
    super(Generator, self).__init__()
    self.size = size
    self.input = SingleEncoder()
    self.condition = Encoder(self.input)
    self.combine = MLP(32, 28 * 28, hidden_size=128, depth=4)

  def sample(self, data):
    support, values = data
    mean, logvar = self.condition(support)
    distribution = Normal(mean, torch.exp(0.5 * logvar))
    latent_sample = distribution.rsample()
    latent_sample = torch.repeat_interleave(latent_sample, self.size, dim=0)
    local_samples = torch.randn(support.size(0) * self.size, 16)
    sample = torch.cat((latent_sample, local_samples), dim=1)
    return (support, sample), (mean, logvar)

  def forward(self, data):
    (support, sample), _ = data
    return support, self.combine(sample).view(-1, self.size, 1, 28, 28).sigmoid()

class Discriminator(nn.Module):
  def __init__(self, encoder, size=5):
    super(Discriminator, self).__init__()
    self.size = size
    self.encoder = encoder
    self.input = encoder.single
    self.verdict = MLP(28 * 28 + 16, 1, hidden_size=128, depth=4, batch_norm=False, activation=func.leaky_relu)

  def forward(self, data):
    support, values = data
    mean, logvar = self.encoder(support)
    distribution = Normal(mean, torch.exp(0.5 * logvar))
    latent_sample = distribution.rsample()
    latent_sample = torch.repeat_interleave(latent_sample, self.size, dim=0)
    combined = torch.cat((values.view(-1, 28 * 28), latent_sample), dim=1)
    return self.verdict(combined)

class MNISTSetTraining(FewShotGANTraining):
  def each_generate(self, data, generated, sample):
    ref = generated[0]
    data = generated[1]
    samples = [sample for sample in ref.contiguous().view(-1, 1, 28, 28)[:10]]
    samples = torch.cat(samples, dim=-1)
    self.writer.add_image("reference", samples, self.step_id)

    samples = [sample for sample in data.view(-1, 1, 28, 28)[:10]]
    samples = torch.cat(samples, dim=-1)
    self.writer.add_image("samples", samples, self.step_id)

if __name__ == "__main__":
  mnist = MNIST("examples/", download=False, transform=ToTensor())
  data = MNISTSet(mnist)

  generator = Generator()
  discriminator = Discriminator(generator.condition)
  
  training = MNISTSetTraining(
    generator, discriminator, data,
    network_name="set-mnist-gan",
    device="cpu",
    batch_size=40,
    max_epochs=1000,
    n_critic=2,
    verbose=True
  )

  training.train()
