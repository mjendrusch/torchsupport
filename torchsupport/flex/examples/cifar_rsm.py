from functools import partial
from torchsupport.modules.backbones.diffusion.unet import DiffusionUNetBackbone, DiffusionUNetBackbone2
from torchsupport.flex.tasks.energy.score_matching import linear_noise
from torchsupport.flex.tasks.energy.relaxed_score_matching import NormalNoise
from torchsupport.flex.training.score_matching import relaxed_score_matching_training
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

def valid_callback(args, ctx: TrainingContext=None):
  ctx.log(images=LogImage(args.sample))
  labels = args.prediction.argmax(dim=1)
  for idx in range(10):
    positive = args.sample[labels == idx]
    if positive.size(0) != 0:
      ctx.log(**{f"classified {idx}": LogImage(positive)})

def generate_step(energy, integrator: Langevin=None, ctx=None):
  sample = 5 * torch.randn(ctx.batch_size, 3, 32, 32, device=ctx.device)
  levels = torch.arange(0.0, 1.0, 0.01, device=ctx.device)
  for level in reversed(levels):
    this_level = level * torch.ones(sample.size(0), device=sample.device)
    sample = integrator.integrate(
      ConditionalEnergy(energy, sample, shift=0.025), sample, this_level, None
    )
  result = ((sample + 1) / 2).clamp(0, 1)
  ctx.log(samples=LogImage(result))

class CIFAR10Dataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __getitem__(self, index):
    data, _ = self.data[index]
    data = data + torch.randn_like(data) / 255
    return 2 * data - 1, []

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
    return torch.randn(batch_size, 3, 32, 32, device=self.mean.device)#dist.rsample(sample_shape=(batch_size,))

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
  def __init__(self):
    super().__init__()
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
    self.W = nn.Linear(256, 256)
    self.b = nn.Linear(256, 1)

  def forward(self, inputs, levels, *args):
    out = inputs
    for res, conv in zip(self.res, self.conv):
      out = func.silu(conv(out))
      out = res(out, levels)
      out = 2 ** 2 * func.avg_pool2d(out, 2)
    features = out.size(-1) ** 2 * func.adaptive_avg_pool2d(out, 1)
    features = features.view(features.size(0), -1)
    quadratic = (features * self.W(features)).sum(dim=1, keepdim=True)
    linear = self.b(features)
    return quadratic + linear

class UNetEnergy(nn.Module):
  def __init__(self):
    super().__init__()
    self.project_in = nn.Conv2d(3, 32, 3, padding=1)
    self.unet = DiffusionUNetBackbone2(
      base=32, factors=[
        [1, 2, 2],
        [2, 4, 4],
        [4, 8, 8]
      ], attention_levels=[1, 2],
      cond_size=128,
      dropout=0.0, norm=False
    )
    self.out = nn.Conv2d(32, 32, 3, padding=1)
    with torch.no_grad():
      for param in self.out.parameters():
        param.zero_()
    self.zero = ReZero(32)
    self.linear = nn.Linear(32, 1)

  def forward(self, inputs, levels, *args):
    out = self.project_in(inputs)
    unet = self.out(func.silu(self.unet(out, levels)))
    # unet = self.zero(torch.zeros_like(unet), unet)
    quadratic = torch.einsum("bchw,bchw->b", unet, out)[:, None]
    linear = self.linear(func.silu(unet).mean(dim=(2, 3)))
    return quadratic# + linear

class TotalEnergy(nn.Module):
  def __init__(self, energy, levels):
    super().__init__()
    self.energy = energy
    self.levels = levels

  def forward(self, data: torch.Tensor, *args):
    inputs = data.repeat_interleave(len(self.levels), dim=0)
    levels = torch.cat(data.size(0) * [self.levels], dim=0)
    factors = self.energy(inputs, levels, *args)
    result = factors.view(-1, data.size(0), 1).sum(dim=0)
    return result

class ConditionalEnergy(nn.Module):
  def __init__(self, energy, origin, shift=0.025):
    super().__init__()
    self.energy = energy
    self.origin = origin.detach()
    self.shift = shift

  def forward(self, data, level, *args):
    raw_energy = self.energy(data, level)
    dist = Normal(self.origin, self.shift)
    cond = dist.log_prob(data)
    cond = cond.view(cond.size(0), -1).mean(dim=1, keepdim=True)
    return raw_energy + cond

if __name__ == "__main__":
  opt = parse_options(
    "CIFAR10 EBM using RSM in flex.",
    path="flexamples/cifar10-rsm-u-19",
    device="cuda:0",
    batch_size=32,
    max_epochs=1000,
    report_interval=1000
  )

  cifar10 = CIFAR10("examples/", download=False, transform=ToTensor())
  data = CIFAR10Dataset(cifar10)
  data = DataDistribution(
    data, batch_size=opt.batch_size,
    device=opt.device
  )

  energy = UNetEnergy().to(opt.device)
  print(energy)

  training = relaxed_score_matching_training(
    energy, data,
    optimizer_kwargs=dict(lr=1e-4),
    level_weight=lambda t: (1e-3 + t * (5.0 - 1e-3)) ** 2,
    level_distribution=NormalNoise(lambda t: 1e-3 + t * (5.0 - 1e-3)),
    noise_distribution=NormalNoise(lambda t: 0.01 * (1e-3 + t * (5.0 - 1e-3))),
    loss_scale=1000,
    path=opt.path,
    device=opt.device,
    batch_size=opt.batch_size,
    max_epochs=opt.max_epochs,
    report_interval=opt.report_interval
  )

  # add generating images every few steps:
  integrator = Langevin(
    rate=-0.1, noise=0.01,
    steps=5, max_norm=None,
    clamp=None
  )
  training.add(
    generate_step=partial(
      generate_step, energy=training.energy_target,
      integrator=integrator,
      ctx=training
    ),
    every=opt.report_interval
  )

  training.load()
  training.train()
