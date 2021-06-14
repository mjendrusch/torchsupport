from functools import partial
from torchsupport.data.namedtuple import namespace

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal
from torch.utils.data import Dataset

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from torchsupport.modules import ReZero
from torchsupport.training.samplers import Langevin
from torchsupport.utils.argparse import parse_options
from torchsupport.flex.log.log_types import LogImage
from torchsupport.flex.context.context import TrainingContext
from torchsupport.flex.data_distributions.data_distribution import DataDistribution
from torchsupport.flex.tasks.energy.density_ratio import direct_mixing, noise_contrastive_estimation, probability_surface_estimation, random_dim_mixing, tdre_mixing, tnce_step, independent_mixing, vp_mixing
from torchsupport.flex.training.density_ratio import telescoping_density_ratio_training

def valid_callback(args, ctx: TrainingContext=None):
  ctx.log(images=LogImage(args.sample))
  labels = args.prediction.argmax(dim=1)
  for idx in range(10):
    positive = args.sample[labels == idx]
    if positive.size(0) != 0:
      ctx.log(**{f"classified {idx}": LogImage(positive)})

def generate_step(energy, base, integrator: Langevin=None, ctx=None):
  sample = base.sample(ctx.batch_size)
  levels = torch.zeros(ctx.batch_size, device=sample.device)
  result = integrator.integrate(energy, sample, levels, None)
  result = result.clamp(0, 1)
  ctx.log(samples=LogImage(result))

class CIFAR10Dataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __getitem__(self, index):
    data, _ = self.data[index]
    data = data + torch.rand_like(data) / 255
    return data, []

  def __len__(self):
    return len(self.data)

class Base(nn.Module):
  def __init__(self):
    super().__init__()
    self.mean = nn.Parameter(torch.zeros(3, 1, 1))
    self.logv = nn.Parameter(torch.zeros(3, 1, 1))

  def sample(self, batch_size):
    dist = Normal(
      self.mean.expand(3, 32, 32),
      self.logv.exp().expand(3, 32, 32)
    )
    return torch.rand(batch_size, 3, 32, 32, device=self.mean.device)#dist.rsample(sample_shape=(batch_size,))

  def log_prob(self, data, condition):
    return torch.zeros_like(self(data, condition)[0])

  def forward(self, data, condition):
    dist = Normal(self.mean, self.logv.exp())
    log_p = dist.log_prob(data)
    log_p = log_p.view(*log_p.shape[:-3], -1)
    return log_p.sum(dim=-1, keepdim=True), namespace(
      distribution=dist
    )

class SineEmbedding(nn.Module):
  def __init__(self, size, depth=2):
    super().__init__()
    self.blocks = nn.ModuleList([
      nn.Linear(1, size)
    ] + [
      nn.Linear(size, size)
      for idx in range(depth - 1)
    ])

  def forward(self, time):
    out = time[:, None]
    for block in self.blocks:
      out = block(out).sin()
    return out

class ResBlock(nn.Module):
  def __init__(self, size):
    super().__init__()
    self.condify = SineEmbedding(2 * size)
    self.skip = SineEmbedding(2 * size)
    self.blocks = nn.ModuleList([
      nn.Conv2d(size, size, 3, padding=1)
      for idx in range(2)
    ])
    self.zero = ReZero(size)

  def forward(self, inputs, levels):
    cond = self.condify(levels)
    cond = cond.view(*cond.shape, 1, 1)
    skip = self.skip(levels)
    skip = skip.view(*skip.shape, 1, 1)
    scale, bias = cond.chunk(2, dim=1)
    skip_scale, skip_bias = skip.chunk(2, dim=1)
    out = func.silu(self.blocks[0](inputs))
    out = scale * out + bias
    out = self.blocks[1](out)
    inputs = skip_scale * inputs + skip_bias
    return self.zero(inputs, out)

class Energy(nn.Module):
  def __init__(self, base):
    super().__init__()
    self.base = base
    self.conv = nn.ModuleList([
      nn.Conv2d(3, 32, 3, padding=1),
      nn.Conv2d(32, 64, 3, padding=1),
      nn.Conv2d(64, 128, 3, padding=1),
      nn.Conv2d(128, 256, 3, padding=1)
    ])
    self.res = nn.ModuleList([
      ResBlock(32),
      ResBlock(64),
      ResBlock(128),
      ResBlock(256),
    ])
    self.out = nn.Linear(256, 1)
    self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
    self.beta = nn.Parameter(torch.ones(1, requires_grad=True))
    self.gamma = nn.Parameter(torch.zeros(1, requires_grad=True))
    self.log_Z = nn.Sequential(
      SineEmbedding(256),
      nn.Linear(256, 1)
    )

  def forward(self, inputs, levels, *args):
    out = inputs
    for res, conv in zip(self.res, self.conv):
      out = func.silu(conv(out))
      out = res(out, levels)
      out = func.avg_pool2d(out, 2)
    features = func.adaptive_avg_pool2d(out, 1)
    raw = self.out(features.view(features.size(0), -1))
    return raw# - self.log_Z(levels)# + self.base.log_prob(inputs, [])

if __name__ == "__main__":
  opt = parse_options(
    "CIFAR10 EBM using TNCE in flex.",
    path="flexamples/cifar10-tnce-54",
    device="cuda:0",
    batch_size=16,
    max_epochs=1000,
    report_interval=1000
  )

  cifar10 = CIFAR10("examples/", download=False, transform=ToTensor())
  data = CIFAR10Dataset(cifar10)
  data = DataDistribution(
    data, batch_size=opt.batch_size,
    device=opt.device
  )

  base = Base().to(opt.device)
  energy = Energy(base).to(opt.device)

  training = telescoping_density_ratio_training(
    energy, base, data,
    mixing=partial(
      independent_mixing,
      mixing=direct_mixing,
      levels=torch.arange(0.0, 1.0, 0.1, device=opt.device)
    ),
    optimizer_kwargs=dict(lr=1e-3),
    telescoping_step=partial(
      tnce_step, noise_contrastive=noise_contrastive_estimation),
    train_base=False,
    path=opt.path,
    device=opt.device,
    batch_size=opt.batch_size,
    max_epochs=opt.max_epochs,
    report_interval=opt.report_interval
  )

  # add generating images every few steps:
  integrator = Langevin(
    rate=-1, noise=0.01,
    steps=100, max_norm=None,
    clamp=None
  )
  training.add(
    generate_step=partial(
      generate_step, energy=energy,
      base=base, integrator=integrator,
      ctx=training
    ),
    every=opt.report_interval
  )
  # training.get_step("tdre_step").extend(
  #   lambda args, ctx=None:
  #     ctx.log(real_images=LogImage(args.real_data.clamp(0, 1)))
  # )

  training.load()
  training.train()
