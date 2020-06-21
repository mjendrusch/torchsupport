import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from torchsupport.modules.basic import MLP
from torchsupport.modules.residual import ResNetBlock2d
from torchsupport.modules.normalization import FilterResponseNorm, NotNorm, AdaNorm, SemiNorm
from torchsupport.training.samplers import Langevin
from torchsupport.interacting.off_ebm import OffEBMTraining
from torchsupport.interacting.energies.energy import Energy
from torchsupport.interacting.shared_data import SharedModule

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
    return (data,)#, label

  def __len__(self):
    return len(self.data)

class Convolutional(nn.Module):
  def __init__(self, depth=4):
    super(Convolutional, self).__init__()
    self.preprocess = spectral_norm(nn.Conv2d(3, 32, 1))
    self.blocks = nn.ModuleList([
      spectral_norm(nn.Conv2d(32, 32, 3, padding=1))
      for idx in range(depth)
    ])
    self.postprocess = spectral_norm(nn.Linear(32, 1))

  def forward(self, inputs):
    out = self.preprocess(inputs)
    for block in self.blocks:
      out = func.relu(out + block(out))
      out = func.avg_pool2d(out, 2)
    out = func.adaptive_avg_pool2d(out, 1).view(-1, 32)
    out = self.postprocess(out)
    return out

class CIFAR10Energy(Energy):
  def prepare(self, batch_size):
    return self.sample_type(
      data=torch.rand(batch_size, 3, 32, 32),
      args=None
    )

class CIFAR10EnergyTraining(OffEBMTraining):
  def each_generate(self, batch):
    data = batch.final_state
    samples = [torch.clamp(sample, 0, 1) for sample in data[0:10]]
    samples = torch.cat(samples, dim=-1)
    self.writer.add_image("samples", samples, self.step_id)

if __name__ == "__main__":
  import torch.multiprocessing as mp
  mp.set_start_method("spawn")

  mnist = CIFAR10("examples/", download=True, transform=ToTensor())
  data = EnergyDataset(mnist)

  score = Convolutional(depth=4)
  energy = CIFAR10Energy(SharedModule(score, dynamic=True), keep_rate=0.95)
  integrator = Langevin(rate=50, steps=20, max_norm=None, clamp=(0, 1))

  training = CIFAR10EnergyTraining(
    score, energy, data,
    network_name="off-energy/cifar10-off-energy-2",
    device="cuda:0",
    integrator=integrator,
    off_energy_weight=5,
    batch_size=64,
    off_energy_decay=1,
    decay=1.0,
    n_workers=8,
    double=True,
    buffer_size=10_000,
    max_steps=int(1e6),
    report_interval=10,
    verbose=True
  )

  training.train()
