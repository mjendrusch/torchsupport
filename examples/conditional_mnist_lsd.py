import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset

from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor

from torchsupport.data.io import make_differentiable
from torchsupport.modules.basic import MLP
from torchsupport.modules.residual import ResNetBlock2d
from torchsupport.modules.normalization import AdaptiveInstanceNormPP, Affine
from torchsupport.training.lsd import LSDTraining
from torchsupport.training.samplers import Langevin

from matplotlib import pyplot as plt

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

class EnergyDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __getitem__(self, index):
    data, label_index = self.data[index]
    data = data + torch.randn_like(data) * 0.1
    label = torch.zeros(10)
    label[label_index] = 1

    #data = -data

    data = (data - data.min()) / (data.max() - data.min())

    result = (data + 1e-6).log() - (1 - data + 1e-6).log()
    #result = data

    return (result, label)

  def __len__(self):
    return len(self.data)

class Critic(nn.Module):
  def __init__(self):
    super(Critic, self).__init__()
    self.input = MLP(
      32 * 32 * 3 + 10, 32 * 32 * 3,
      hidden_size=512,
      depth=4,
      batch_norm=False,
      activation=swish
    )
    self.embed = nn.Linear(10, 10)

  def forward(self, image, cond):
    image = image.view(-1, 32 * 32 * 3)
    cond = self.embed(cond)
    result = self.input(torch.cat((image, cond), dim=1))
    return result.view(result.size(0), 3, 32, 32)

def swish(x):
  return x * x.sigmoid()

the_norm = lambda x: x

class ConvCritic(nn.Module):
  def __init__(self):
    super(ConvCritic, self).__init__()
    self.input = the_norm(nn.Conv2d(3, 128, 3, padding=1))
    self.out = the_norm(nn.Conv2d(128, 3, 3, padding=1))
    self.down = nn.ModuleList([
      the_norm(nn.Conv2d(128, 128, 3, padding=1))
      for idx in range(3)
    ])
    self.up = nn.ModuleList([
      the_norm(nn.Conv2d(128 + 128, 128, 3, padding=1))
      for idx in range(3)
    ])
    self.center = the_norm(nn.Conv2d(128, 128, 3, padding=1))

  def forward(self, inputs, cond):
    out = self.input(inputs)
    results = []
    for block in self.down:
      out = block(swish(out))
      results = [out] + results
      out = func.avg_pool2d(out, 2)
    out = self.center(swish(out))
    for skip, block in zip(results, self.up):
      out = func.interpolate(out, scale_factor=2, mode="bilinear")
      out = block(swish(torch.cat((out, skip), dim=1)))
    out = self.out(swish(out))
    return out

class ConvEnergy(nn.Module):
  def __init__(self):
    super(ConvEnergy, self).__init__()
    self.input = the_norm(nn.Conv2d(3, 32, 3, padding=1))
    self.blocks = nn.ModuleList([
      the_norm(nn.Conv2d(32, 32, 3, padding=1))
      for idx in range(5)
    ])
    self.bn = nn.ModuleList([
     Affine(32, 10)
     for idx in range(6)
    ])
    self.predict = the_norm(nn.Linear(32, 1))
    self.mean = the_norm(nn.Linear(10, 1, bias=False))
    self.logvar = the_norm(nn.Linear(10, 1, bias=False))

  def forward(self, inputs, cond):
    #out = swish(self.input(inputs))
    #for idx, (bn, block) in enumerate(zip(self.bn, self.blocks)):
    #  out = out + 0.1 * swish(block(out))
    #  if idx % 2 == 0:
    #    out = func.avg_pool2d(out, 2)
    #energy = self.predict(func.adaptive_avg_pool2d(out, 1).view(out.size(0), -1))
    energy = 0.0
    gauss = ((inputs.view(-1, 3 * 32 * 32) - self.mean(cond)) ** 2 / (self.logvar(cond).exp() + 1e-6)).mean(dim=-1)
    return -(energy + gauss)

class UNetEnergy(ConvCritic):
  def __init__(self):
    super().__init__()
    self.logv = nn.Parameter(torch.zeros(1, requires_grad=True))
    self.pmean = nn.Parameter(torch.zeros(3, requires_grad=True))
    self.plogv = nn.Parameter(torch.zeros(3, requires_grad=True))

  def forward(self, inputs, cond):
    out = super().forward(inputs, cond)
    off = ((inputs - self.pmean[None, :, None, None]) ** 2) / self.plogv[None, :, None, None].exp()
    off = -off.view(inputs.size(0), -1).mean(dim=1)
    dist = Normal(out, self.logv[0].exp())
    return dist.log_prob(inputs).view(inputs.size(0), -1).mean(dim=1) + off

class Energy(nn.Module):
  def __init__(self):
    super(Energy, self).__init__()
    self.input = MLP(
      32 * 32 * 3 + 10, 1,
      hidden_size=1000,
      depth=5,
      batch_norm=False,
      activation=swish
    )
    self.embed = nn.Linear(10, 10)
    self.mean = nn.Linear(10, 1, bias=False)
    self.logvar = nn.Linear(10, 1, bias=False)

    with torch.no_grad():
      self.mean.weight.zero_()
      self.logvar.weight.zero_()

  def forward(self, image, cond):
    image = image.view(-1, 32 * 32 * 3)
    cond = self.embed(cond)
    result = self.input(torch.cat((image, cond), dim=1))
    gauss = ((image - self.mean(cond)) ** 2 / (self.logvar(cond).exp() + 1e-6)).mean(dim=-1)
    return -(result + gauss)

class MNISTEnergyTraining(LSDTraining):
  def prepare(self):
    data = torch.rand(3, 32, 32)

    label_index = torch.randint(0, 10, (1,))[0]
    label = torch.zeros(10)
    label[label_index] = 1

    #data = -data
    data = data + torch.rand_like(data) / 255

    data = (data - data.min()) / (data.max() - data.min())

    result = torch.randn_like(data)
    #result = data
    return (result, label)

  def each_generate(self, data, *args):
    noise = args[0]
    samples = [sample for sample in noise[0:10]]
    samples = torch.cat(samples, dim=-1).sigmoid()
    self.writer.add_image("noise", samples, self.step_id)

    samples = [sample for sample in data[0:10]]
    samples = torch.cat(samples, dim=-1).sigmoid()
    self.writer.add_image("samples", samples, self.step_id)

if __name__ == "__main__":
  mnist = CIFAR10("examples/", download=False, transform=ToTensor())
  data = EnergyDataset(mnist)

  energy = UNetEnergy()
  critic = ConvCritic()

  training = MNISTEnergyTraining(
    energy, critic, data,
    network_name="runs/experimental/learned-stein/conv-1",
    device="cuda:0",
    batch_size=64,
    decay=10.0,
    report_interval=1000,
    max_epochs=1000,
    optimizer_kwargs={"lr": 1e-4, "betas": (0.0, 0.9)},
    critic_optimizer_kwargs={"lr": 1e-4, "betas": (0.0, 0.9)},
    n_critic=5,
    integrator=Langevin(rate=-0.01, steps=100, noise=0.01, clamp=(-14, 14), max_norm=None),
    verbose=True
  )

  training.train()
