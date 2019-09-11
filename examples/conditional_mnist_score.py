import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from torchsupport.modules.basic import MLP
from torchsupport.modules.residual import ResNetBlock2d
from torchsupport.modules.normalization import AdaptiveInstanceNormPP
from torchsupport.training.energy import DenoisingScoreTraining

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

class EnergyDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __getitem__(self, index):
    data, label_index = self.data[index]
    data = data * 255 / 256 + torch.randn_like(data) / 256
    label = torch.zeros(10)
    label[label_index] = 1
    return (data,)

  def __len__(self):
    return len(self.data)

class Energy(nn.Module):
  def __init__(self):
    super(Energy, self).__init__()
    self.input = MLP(28 * 28 + 10, 28 * 28, hidden_size=128, depth=4, batch_norm=False, activation=func.elu)

  def forward(self, image, noise):
    image = image * 2 - 1
    cond = torch.zeros(image.size(0), 10)
    offset = (torch.log(noise) / torch.log(torch.tensor(0.60))).long()
    cond[torch.arange(image.size(0)), offset.view(-1)] = 1
    image = image.view(-1, 28 * 28)
    result = self.input(torch.cat((image, cond), dim=1))
    return result.view(result.size(0), 1, 28, 28)

class UNetEnergy(nn.Module):
  def __init__(self, depth=3):
    super(UNetEnergy, self).__init__()
    self.input = nn.Conv2d(1, 16, 3, padding=1)
    self.output = nn.Conv2d(16, 1, 1)
    self.down = nn.ModuleList([
      nn.Conv2d(
        16 * 2 ** idx, 16 * 2 ** (idx + 1), 3,
        dilation=2 ** idx, padding=2 ** idx
      )
      for idx in range(depth)
    ])
    self.down_norm = nn.ModuleList([
      AdaptiveInstanceNormPP(16 * 2 ** idx, 10)
      for idx in range(depth)
    ])
    self.up = nn.ModuleList([
      nn.Conv2d(
        2 * 16 * 2 ** (idx + 1), 16 * 2 ** idx, 3,
        dilation=2 ** (idx + 1), padding=2 ** (idx + 1)
      )
      for idx in reversed(range(depth))
    ])
    self.up_norm = nn.ModuleList([
      AdaptiveInstanceNormPP(2 * 16 * 2 ** (idx + 1), 10)
      for idx in reversed(range(depth))
    ])

  def forward(self, inputs, noise):
    out = self.input(inputs)
    cond = torch.zeros(inputs.size(0), 10)
    offset = (torch.log(noise) / torch.log(torch.tensor(0.60))).long()
    cond[torch.arange(inputs.size(0)), offset.view(-1)] = 1
    connections = []
    for norm, block in zip(self.down_norm, self.down):
      out = func.relu(block(norm(out, cond)))
      connections.append(out)
    for norm, block, shortcut in zip(self.up_norm, self.up, reversed(connections)):
      out = func.relu(block(norm(torch.cat((out, shortcut), dim=1), cond)))
    del connections
    return self.output(out)    

class MNISTEnergyTraining(DenoisingScoreTraining):
  def prepare(self):
    return (torch.rand(1, 28, 28),)

  def each_generate(self, data, *args):
    noise = args[0]
    samples = [sample for sample in noise[0:10]]
    samples = torch.cat(samples, dim=-1)
    self.writer.add_image("noise", samples, self.step_id)

    samples = [sample for sample in data[0:10]]
    samples = torch.cat(samples, dim=-1)
    self.writer.add_image("samples", samples, self.step_id)

if __name__ == "__main__":
  mnist = MNIST("examples/", download=False, transform=ToTensor())
  data = EnergyDataset(mnist)

  energy = UNetEnergy()
  
  training = MNISTEnergyTraining(
    energy, data,
    network_name="conditional-mnist-ebm",
    device="cpu",
    batch_size=16,
    optimizer_kwargs={"lr": 0.001},
    report_steps=100,
    max_epochs=1000,
    verbose=True
  )

  training.train()
