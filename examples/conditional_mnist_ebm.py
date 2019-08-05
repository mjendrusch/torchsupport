import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from torchsupport.modules.basic import MLP
from torchsupport.modules.residual import ResNetBlock2d
from torchsupport.training.energy import EnergyTraining, Langevin

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

class EnergyDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __getitem__(self, index):
    data, label_index = self.data[index]
    # data = data + 0.05 * torch.rand_like(data)
    label = torch.zeros(10)
    label[label_index] = 1
    return data, label

  def __len__(self):
    return len(self.data)

class Energy(nn.Module):
  def __init__(self):
    super(Energy, self).__init__()
    self.input = MLP(28 * 28, 128, hidden_size=128, depth=3, batch_norm=False, normalization=spectral_norm)
    self.condition = MLP(10, 128, depth=3, batch_norm=False, normalization=spectral_norm)
    self.combine = MLP(128, 1, hidden_size=64, depth=3, batch_norm=False, normalization=spectral_norm)

  def forward(self, image, condition):
    image = image.view(-1, 28 * 28)
    out = self.input(image)
    cond = self.condition(condition)
    result = self.combine(func.relu(out + cond))
    return result

class ConvEnergy(nn.Module):
  def __init__(self, depth=4):
    super(ConvEnergy, self).__init__()
    self.preprocess = nn.Conv2d(1, 32, 1)
    self.blocks = nn.ModuleList([
      spectral_norm(nn.Conv2d(32, 32, 3, padding=1))
      for idx in range(depth)
    ])
    self.bn = nn.ModuleList([
      nn.GroupNorm(8, 32)
      for idx in range(depth)
    ])
    self.postprocess = nn.Conv2d(32, 128, 1)
    self.condition = MLP(10, 128, depth=3, batch_norm=False, normalization=spectral_norm)
    self.combine = MLP(128, 1, hidden_size=64, depth=3, batch_norm=False, normalization=spectral_norm)

  def forward(self, inputs, condition):
    out = self.preprocess(inputs)
    for bn, block in zip(self.bn, self.blocks):
      out = func.relu(bn(out + block(out)))
      out = func.avg_pool2d(out, 2)
    out = self.postprocess(out)
    out = func.adaptive_avg_pool2d(out, 1).view(-1, 128)
    cond = self.condition(condition)
    result = self.combine(func.relu(out + cond))
    return result

class MNISTEnergyTraining(EnergyTraining):
  def each_generate(self, data, args):
    samples = [sample for sample in data[0:10]]
    samples = torch.cat(samples, dim=-1)
    self.writer.add_image("samples", samples, self.step_id)

if __name__ == "__main__":
  mnist = MNIST("examples/", download=False, transform=ToTensor())
  data = EnergyDataset(mnist)

  energy = ConvEnergy()
  integrator = Langevin(rate=30, steps=50, max_norm=None)
  
  training = MNISTEnergyTraining(
    energy, data,
    network_name="conditional-mnist-ebm",
    device="cpu",
    integrator=integrator,
    batch_size=64,
    max_epochs=1000,
    verbose=True
  )

  training.train()
