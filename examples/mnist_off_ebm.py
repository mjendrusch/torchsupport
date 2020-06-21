import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset

from torchvision.datasets import MNIST
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
    self.preprocess = spectral_norm(nn.Conv2d(1, 32, 1))
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

class Connected(nn.Module):
  def __init__(self, depth=4):
    super(Connected, self).__init__()
    self.block = MLP(
      28 * 28, 1,
      depth=depth,
      batch_norm=False,
      normalization=spectral_norm
    )

  def forward(self, inputs):
    out = inputs.view(inputs.size(0), -1)
    out = self.block(out)
    return out

class MNISTEnergy(Energy):
  def prepare(self, batch_size):
    return self.sample_type(
      data=torch.rand(batch_size, 1, 28, 28),
      args=None
    )

class MNISTEnergyTraining(OffEBMTraining):
  def each_generate(self, batch):
    data = batch.final_state
    samples = [torch.clamp(sample, 0, 1) for sample in data[0:10]]
    samples = torch.cat(samples, dim=-1)
    self.writer.add_image("samples", samples, self.step_id)

if __name__ == "__main__":
  import torch.multiprocessing as mp
  mp.set_start_method("spawn")

  mnist = MNIST("examples/", download=False, transform=ToTensor())
  data = EnergyDataset(mnist)

  score = Convolutional(depth=4)
  energy = MNISTEnergy(SharedModule(score, dynamic=False), keep_rate=0.95)
  integrator = Langevin(rate=10, steps=10, noise=0.01, max_norm=None, clamp=(0, 1))

  training = MNISTEnergyTraining(
    score, energy, data,
    network_name="off-energy/mnist-conv-off-e-loss-8",
    device="cuda:0",
    integrator=integrator,
    off_energy_weight=0,
    batch_size=64,
    optimizer_kwargs=dict(lr=1e-4, betas=(0.0, 0.999)),
    decay=1.0,
    n_workers=8,
    double=True,
    buffer_size=1_000,
    max_steps=int(1e6),
    report_interval=10,
    verbose=True
  )

  training.train()
