import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from torchsupport.modules.basic import MLP
from torchsupport.modules.residual import ResNetBlock2d
from torchsupport.modules.normalization import FilterResponseNorm
from torchsupport.training.samplers import Langevin, AdaptiveLangevin
from torchsupport.training.energy_supervised import (
  EnergySupervisedTraining, EnergyConditionalTraining
)

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

class EnergyDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __getitem__(self, index):
    data, label_index = self.data[index]
    data = 2 * data - 1 + 0.03 * torch.randn_like(data)
    return data, label_index

  def __len__(self):
    return len(self.data)

def _next(idx):
  return 32# * 2 ** (idx // 3)

def upscale(size):
  def _inner(data):
    current_size = data.size(1)
    difference = size - data.size(1)
    if difference > 0:
      data = torch.cat((
        data,
        torch.zeros(
          data.shape[0], difference, *data.shape[2:],
          device=data.device, dtype=data.dtype
        )
      ), dim=1)
    return data
  return _inner

class ConvEnergy(nn.Module):
  def __init__(self, depth=4):
    super(ConvEnergy, self).__init__()
    self.preprocess = nn.Conv2d(3, _next(0), 1)
    self.blocks = nn.ModuleList([
      spectral_norm(nn.Conv2d(_next(idx), _next(idx + 1), 3, padding=1))
      for idx in range(depth)
    ])
    self.project = [
      upscale(_next(idx + 1))
      for idx in range(depth)
    ]
    self.bn = nn.ModuleList([
      nn.ReLU()
      for idx in range(depth)
    ])
    self.postprocess = spectral_norm(nn.Conv2d(_next(depth), 128, 1))
    self.predict = spectral_norm(nn.Linear(128, 10))

  def forward(self, inputs, *args):
    out = self.preprocess(inputs)
    count = 0
    for bn, proj, block in zip(self.bn, self.project, self.blocks):
      out = bn(proj(out) + block(out))
      count += 1
      if count % 5 == 0:
        out = func.avg_pool2d(out, 2)
    out = self.postprocess(out)
    out = func.adaptive_avg_pool2d(out, 1).view(-1, 128)
    return self.predict(out)

class Classifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.energy = nn.Parameter(torch.randn(1))#MLP(10, 1, 64, depth=3, normalization=spectral_norm, batch_norm=False)
    self.predict = MLP(28 * 28, 10, 128, depth=3, normalization=spectral_norm, batch_norm=False)

  def forward(self, inputs, *args):
    return self.predict(inputs.view(inputs.size(0), -1))

class CIFAR10EnergyTraining(EnergySupervisedTraining):
  def prepare(self):
    data = torch.rand(3, 32, 32)
    return (2 * data - 1,)

  def data_key(self, data):
    if isinstance(data, (list, tuple)):
      return data
    else:
      return (data,)

  def each_generate(self, data):
    samples = [(torch.clamp(sample, -1, 1) + 1) / 2 for sample in data[0:10]]
    samples = torch.cat(samples, dim=-1)
    self.writer.add_image("samples", samples, self.step_id)

if __name__ == "__main__":
  mnist = CIFAR10("examples/", download=False, transform=ToTensor())
  data = EnergyDataset(mnist)

  energy = ConvEnergy(depth=20)
  integrator = Langevin(rate=1, noise=0.01, steps=20, clamp=None, max_norm=None)

  training = CIFAR10EnergyTraining(
    energy, data,
    network_name="classifier-mnist-ebm/cifar-plain",
    device="cuda:0",
    integrator=integrator,
    decay=0.0,
    batch_size=16,
    buffer_size=10000,
    optimizer_kwargs={"lr": 1e-4},
    max_epochs=1000,
    verbose=True
  ).load()

  training.train()
