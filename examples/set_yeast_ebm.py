import os
import sys
import random

import numpy as np
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
from torchsupport.training.samplers import Langevin
from torchsupport.training.energy import SetVAETraining, CyclicSetVAETraining

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

class YeastSet(Dataset):
  def __init__(self, path, size=5):
    self.size = size
    self.files = [
      os.path.join(root, fpath)
      for root, directories, fpaths in os.walk(path)
      for fpath in fpaths
      if fpath.endswith("ome.tif")
    ]
    self.files = self.files[:1000]# FIXME
    self.cache = np.zeros((len(self.files), 3, 975, 975))
    idx = 0
    for file_name in self.files:
      try:
        image = imread(file_name).numpy()
        if image.shape != (3, 975, 975):
          print(image.shape)
          continue
        self.cache[idx] = image
        idx += 1
      except KeyboardInterrupt as e:
        raise e
      except Exception as e:
        print(*e.args)
    self.cache = self.cache[:idx]

  def __getitem__(self, index):
    done = False
    pickfile = random.randrange(len(self.cache))
    result = []
    data = torch.tensor(self.cache[pickfile], dtype=torch.float)
    offset = 1
    size = 65
    for idx in range(self.size):
      x = torch.randint(0, 15, (1,))[0]
      y = torch.randint(0, 15, (1,))[0]
      pick = data[:, x * size + offset:x * size + offset + 64, y * size + offset:y * size + offset + 64]
      pick = pick.contiguous()
      pick = pick - pick.view(3, -1).min(dim=-1).values.view(3, 1, 1)
      pick = pick / (pick.view(3, -1).max(dim=-1).values.view(3, 1, 1) + 1e-6)
      result.append(pick.unsqueeze(0))
    result = torch.cat(result, dim=0)
    return result, result

  def __len__(self):
    return 100 * len(self.files)

class FixUpRes(nn.Module):
  def __init__(self, in_size=128, out_size=128, branches=1):
    super(FixUpRes, self).__init__()
    self.conv_0 = spectral_norm(nn.Conv2d(in_size, in_size // 2, 3, padding=1, bias=False))
    self.conv_1 = spectral_norm(nn.Conv2d(in_size // 2, out_size, 3, padding=1, bias=False))
    self.project = spectral_norm(nn.Conv2d(in_size, out_size, 1, bias=False))
    self.register_parameter("b_0", nn.Parameter(0.0 * torch.randn(1)[0]))
    self.register_parameter("b_1", nn.Parameter(0.0 * torch.randn(1)[0]))
    self.register_parameter("b_2", nn.Parameter(0.0 * torch.randn(1)[0]))
    self.register_parameter("b_3", nn.Parameter(0.0 * torch.randn(1)[0]))
    self.register_parameter("m_0", nn.Parameter(torch.ones(1)[0]))
    with torch.no_grad():
      self.conv_0.weight /= torch.sqrt(torch.tensor(branches, dtype=torch.float))
      self.conv_1.weight = 1e-10 * torch.randn_like(self.conv_1.weight)

  def forward(self, inputs):
    out = self.conv_0(inputs + self.b_0) + self.b_1
    out = func.relu(out)
    out = self.conv_1(out + self.b_2) * self.m_0 + self.b_3
    return func.relu(out + self.project(inputs))

class SingleEncoder(nn.Module):
  def __init__(self, latents=128):
    super(SingleEncoder, self).__init__()
    self.input = nn.Conv2d(3, 128, 3, padding=1)
    self.blocks = nn.ModuleList([
      #FixUpRes(32 * 2 ** (idx // 2), 32 * 2 ** ((idx + 1) // 2), 9)
      spectral_norm(nn.Conv2d(128, 128, 3, padding=1))
      for idx in range(9)
    ])
    #self.out = spectral_norm(nn.Linear(32 * 2 ** (9 // 2), latents))
    self.out = MLP(
      128, latents, hidden_size=64,
      depth=3, batch_norm=False,
      normalization=spectral_norm,
      activation=func.leaky_relu
    )

  def forward(self, inputs):
    out = func.leaky_relu(self.input(inputs))
    for idx, block in enumerate(self.blocks):
      out = func.leaky_relu(block(out))
      if idx % 2 == 0:
        out = func.avg_pool2d(out, 2)
    out = func.adaptive_avg_pool2d(out, 1).view(-1, 128)#, 32 * 2 ** (9 // 2))
    return self.out(out)

class Encoder(nn.Module):
  def __init__(self, single, size=5, latents=64):
    super(Encoder, self).__init__()
    self.size = size
    self.single = single
    self.weight = spectral_norm(nn.Linear(128, 1))
    self.combine = MLP(128, 128, 64, depth=3, batch_norm=False, normalization=spectral_norm, activation=func.leaky_relu)
    self.mean = spectral_norm(nn.Linear(128, latents))
    self.logvar = spectral_norm(nn.Linear(128, latents))

  def forward(self, inputs):
    inputs = inputs.view(-1, 3, 64, 64)
    out = self.single(inputs)
    weights = self.weight(out)
    out = out.view(-1, self.size, 128)
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
    self.input_process = spectral_norm(nn.Linear(128, 64))
    self.postprocess = spectral_norm(nn.Linear(64, 64, bias=False))
    self.combine = MLP(128, 1, hidden_size=64, depth=4, batch_norm=False, normalization=spectral_norm, activation=func.leaky_relu)

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
    self.writer.add_image("reference", samples.unsqueeze(0), self.step_id)

    samples = [sample for sample in data.view(-1, 3, 64, 64)[:10]]
    samples = torch.cat(samples, dim=-1)
    samples = torch.cat([point for point in samples], dim=-2)
    self.writer.add_image("samples", samples.unsqueeze(0), self.step_id)

if __name__ == "__main__":
  data = YeastSet(sys.argv[1])

  energy = nn.DataParallel(Energy())
  integrator = Langevin(rate=100, steps=40, max_norm=None)
  
  training = YeastSetTraining(
    energy, data,
    network_name="set-yeast-plain-ae",
    device="cuda:0",
    oos_penalty=False,
    integrator=integrator,
    buffer_probability=0.95,
    buffer_size=10000,
    batch_size=20,
    max_epochs=1000,
    optimizer_kwargs={"lr": 1e-4, "betas": (0.0, 0.999)},
    verbose=True
  )

  training.train()
