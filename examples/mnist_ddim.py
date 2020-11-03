import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset

from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor

from torchsupport.modules.basic import MLP
from torchsupport.training.samplers import Langevin
from torchsupport.modules.unet import UNetBackbone, IgnoreArgs
from torchsupport.training.denoising_diffusion import DenoisingDiffusionTraining
from torchsupport.modules.attention import NonLocal

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

class EnergyDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __getitem__(self, index):
    data, label_index = self.data[index]
    data = data + (torch.rand_like(data) - 0.5) / 256
    data = data.clamp(0.001, 0.999)
    data = 2 * data - 1
    label = torch.zeros(10)
    label[label_index] = 1
    return (data,)

  def __len__(self):
    return len(self.data)

class Denoiser(nn.Module):
  def __init__(self):
    super().__init__()
    self.input = MLP(
      28 * 28 + 100, 28 * 28,
      hidden_size=128, depth=3,
      batch_norm=False
    )

  def time_embedding(self, time):
    time = time.float()[:, None]
    return torch.cat([
      (time / (idx + 1)).sin()
      for idx in range(100)
    ], dim=1)

  def forward(self, inputs, time):
    inputs = inputs.view(-1, 28 * 28)
    time = self.time_embedding(time)
    inputs = torch.cat((inputs, time), dim=1)
    out = self.input(inputs)
    return out.view(-1, 1, 28, 28)

class UDenoiser(nn.Module):
  def __init__(self, in_size=3):
    super().__init__()
    self.project = nn.Conv2d(in_size, 64, 7, padding=3)
    self.bb = UNetBackbone(
      size_factors=[1, 1, 2, 2], activation=swish,
      base_size=64, kernel_size=5, depth=2,
      cond_size=100, hidden_size=64,
      norm=nn.InstanceNorm2d,
      hole=IgnoreArgs(NonLocal(2 * 64))
    )
    self.predict_bn = nn.InstanceNorm2d(64 * 2)
    self.predict = nn.Conv2d(64, in_size, 1)

  def time_embedding(self, time):
    time = time.float()[:, None]
    return torch.cat([
      (time / (1000 ** (idx / 100))).sin()
      for idx in range(50)
    ] + [
      (time / (1000 ** (idx / 100))).cos()
      for idx in range(50)
    ], dim=1)

  def forward(self, inputs, time):
    time = self.time_embedding(time)
    out = self.project(inputs)
    out = self.bb(out, time)
    out = self.predict_bn(out)
    out = self.predict(out)
    return out

def swish(x):
  return x * x.sigmoid()

if __name__ == "__main__":
  mnist = CIFAR10("examples/", download=False, transform=ToTensor())
  data = EnergyDataset(mnist)

  denoiser = UDenoiser()

  training = DenoisingDiffusionTraining(
    denoiser, data,
    network_name="mnist-ddim/CIFAR-1",
    timesteps=1000,
    skipsteps=1,
    optimizer_kwargs=dict(lr=2e-4),
    device="cuda:0",
    batch_size=2,
    max_epochs=1000,
    report_interval=1000,
    checkpoint_interval=5000,
    verbose=True
  )

  training.train()
