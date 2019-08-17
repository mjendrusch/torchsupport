import os
import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset
from torch.distributions import Normal

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from torchsupport.data.io import imread
from torchsupport.modules.basic import MLP
from torchsupport.modules.residual import ResNetBlock2d
from torchsupport.training.energy import SetVAETraining, Langevin

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

class YeastSet(Dataset):
  def __init__(self, path, size=5):
    self.size = size
    self.files = [
      os.path.join(root, fpath)
      for root, directory, fpath in os.walk(path)
      if fpath.endswith("ome.tif")
    ]

  def __getitem__(self, index):
    pickfile = random.randrange(len(self.files))
    data = imread(self.files[pickfile])
    offset = 1
    size = 65
    for idx in range(self.size):
      x = torch.randint(0, 15, (1,))[0]
      y = torch.randint(0, 15, (1,))[0]
      pick = data[:, x * size + offset, y * size + offset]
      data.append(pick.unsqueeze(0))
    data = torch.cat(data, dim=0)
    return data, data

class FixUpRes(nn.Module):
  def __init__(self, size=128, branches=1):
    super(FixUpRes, self).__init__()
    self.conv_0 = spectral_norm(nn.Conv2d(size, size // 2, 3, padding=1))
    self.conv_1 = spectral_norm(nn.Conv2d(size // 2, size, 3, padding=1))
    self.register_parameter("b_0", 1e-10 * torch.randn(1)[0])
    self.register_parameter("b_1", 1e-10 * torch.randn(1)[0])
    self.register_parameter("b_2", 1e-10 * torch.randn(1)[0])
    self.register_parameter("b_3", 1e-10 * torch.randn(1)[0])
    self.register_parameter("m_0", torch.ones(1)[0])
    with torch.no_grad():
      self.conv_0.weight /= torch.sqrt(torch.tensor(2 ** branches))
      self.conv_1.weight = 1e-10 * torch.randn()

  def forward(self, inputs):
    out = self.conv_0(inputs + self.b_0) + self.b_1
    out = func.relu(out)
    out = self.conv_1(out + self.b_2) * self.m_0 + self.b_3
    return func.relu(out + inputs)

class SingleEncoder(nn.Module):
  def __init__(self, latents=32):
    super(SingleEncoder, self).__init__()
    self.input = nn.Conv2d(3, 128, 3, padding=1)
    self.blocks = nn.ModuleList([
      FixUpRes(128, 3)
      for idx in range(6)
    ])
    self.out = MLP(
      128, latents, hidden_size=64,
      depth=2, batch_norm=False,
      normalization=spectral_norm
    )

  def forward(self, inputs):
    out = func.relu(self.input(inputs))
    for idx, block in enumerate(self.blocks):
      out = block(out)
      if idx % 2 == 0:
        out = func.avg_pool2d(out, 2)
    out = func.adaptive_avg_pool2d(out, 1).view(-1, 128)
    return self.out(out)

class Encoder(nn.Module):
  def __init__(self, single, size=5, latents=16):
    super(Encoder, self).__init__()
    self.size = size
    self.single = single
    self.weight = spectral_norm(nn.Linear(32, 1))
    self.combine = MLP(32, 32, 64, depth=3, batch_norm=False, normalization=spectral_norm)
    self.mean = spectral_norm(nn.Linear(32, latents))
    self.logvar = spectral_norm(nn.Linear(32, latents))

  def forward(self, inputs):
    inputs = inputs.view(-1, 3, 64, 64)
    out = self.single(inputs)
    weights = self.weight(out)
    out = out.view(-1, self.size, 32)
    weights = weights.view(-1, self.size, 1).softmax(dim=1)
    pool = (weights * out).sum(dim=1)
    pool = self.combine(pool)
    return self.mean(pool), self.logvar(pool)

class Energy(nn.Module):
  def __init__(self, sample=True):
    super(Energy, self).__init__()
    self.sample = sample

    self.input = SingleEncoder()
    self.condition = Encoder(self.input)
    self.input_process = spectral_norm(nn.Linear(32, 64))
    self.postprocess = spectral_norm(nn.Linear(16, 64))
    self.combine = MLP(128, 1, hidden_size=64, depth=4, batch_norm=False, normalization=spectral_norm)

  def forward(self, image, condition):
    image = image.view(-1, 3, 64, 64)
    out = self.input_process(self.input(image))
    mean, logvar = self.condition(condition)
    #distribution = Normal(mean, torch.exp(0.5 * logvar))
    sample = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)#distribution.rsample()
    cond = self.postprocess(sample)
    cond = torch.repeat_interleave(cond, 5, dim=0)
    result = self.combine(torch.cat((out, cond), dim=1))
    return result, (mean, logvar)

class YeastSetTraining(SetVAETraining):
  def each_generate(self, data, *args):
    ref = args[0]
    samples = [sample for sample in ref.contiguous().view(-1, 3, 64, 64)[:10]]
    samples = torch.cat(samples, dim=-1)
    samples = torch.cat([point for point in samples], dim=-2)
    self.writer.add_image("reference", samples, self.step_id)

    samples = [sample for sample in data.view(-1, 3, 64, 64)[:10]]
    samples = torch.cat(samples, dim=-1)
    samples = torch.cat([point for point in samples], dim=-2)
    self.writer.add_image("samples", samples, self.step_id)

if __name__ == "__main__":
  data = YeastSet(sys.argv[1])

  energy = Energy()
  integrator = Langevin(rate=30, steps=30, max_norm=None)
  
  training = YeastSetTraining(
    energy, data,
    network_name="set-mnist-reg-noisy",
    device="cuda:0",
    integrator=integrator,
    buffer_probability=0.95,
    buffer_size=10000,
    batch_size=40,
    max_epochs=1000,
    verbose=True
  )

  training.train()
