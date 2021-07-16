from functools import partial
from torchsupport.modules.backbones.diffusion.unet import DiffusionUNetBackbone, DiffusionUNetBackbone2, ExternalAttentionBlock, AttentionBlock
from torchsupport.flex.tasks.energy.score_matching import linear_noise
from torchsupport.flex.tasks.energy.relaxed_score_matching import NormalNoise, TruncatedNormalNoise, LangevinNoise, RecoveryNoise
from torchsupport.flex.training.score_matching import recovery_likelihood_training
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
from torchsupport.structured.modules.attention import cross_attention, cross_low_rank_mvp, cross_gated_mvp
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
    data = (255 * data + torch.rand_like(data)) / 256
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
    size = 128
    self.project_in = nn.Conv2d(3, size, 3, padding=1)
    self.unet = DiffusionUNetBackbone2(
      base=size, factors=[
        [1, 1, 2],
        [2, 2, 2],
        [2, 2, 2]
      ], attention_levels=[],
      cond_size=4 * size,
      dropout=0.1, norm=True,
      middle_attention=False,
      kernel=cross_low_rank_mvp,
      attention_block=AttentionBlock
    )
    self.out = nn.Conv2d(size, 3, 3, padding=1)
    #with torch.no_grad():
    #  for param in self.out.parameters():
    #    param.zero_()

  def forward(self, inputs, levels, *args):
    #sigmas = 0.05 + levels * (10 - 0.05)
    #inputs = (1 / (sigmas + 1))[:, None, None, None] * inputs
    out = self.project_in(inputs)
    unet = self.out(func.silu(self.unet(out, levels)))
    #out = inputs - unet
    out = ((inputs - unet).view(out.size(0), -1) ** 2).sum(dim=1, keepdim=True)
    quadratic = torch.einsum("bchw,bchw->b", out, out)[:, None]
    factor = 1 / (1e-3 + levels * (5 - 1e-3))
    return -quadratic * factor

class AdaptedUNetEnergy(nn.Module):
  def __init__(self):
    super().__init__()
    size = 128
    self.project_in = nn.Conv2d(3, size, 3, padding=1)
    self.unet = DiffusionUNetBackbone2(
      base=size, factors=[
        [1, 1, 2],
        [2, 2, 2],
        [2, 2, 2]
      ], attention_levels=[1, 2],
      cond_size=4 * size,
      dropout=0.1, norm=True,
      middle_attention=True,
      kernel=cross_attention,
      attention_block=AttentionBlock
    )
    self.out = nn.Conv2d(size, 3, 3, padding=1)
    with torch.no_grad():
      for param in self.out.parameters():
        param.zero_()
    self.factor = nn.Parameter(torch.zeros(1, requires_grad=True))

  def forward(self, inputs, levels, *args):
    #sigmas = 0.05 + levels * (10 - 0.05)
    #inputs = (1 / (sigmas + 1))[:, None, None, None] * inputs
    out = self.project_in(inputs)
    unet = self.out(func.silu(self.unet(out, levels)))
    #unet = unet + inputs # ensures same init as score matching unet!
    out = inputs - unet
    #quadratic = ((inputs - unet).view(out.size(0), -1) ** 2).sum(dim=1, keepdim=True)
    quadratic = torch.einsum("bchw,bchw->b", out, out)[:, None]
    factor = 1 / (2 * (1e-3 + levels * (5 - 1e-3)) ** 2)
    result = -quadratic# * factor
    return result * self.factor[0].exp()

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
    #dist = Normal(self.origin, self.shift)
    #cond = dist.log_prob(data)
    #cond = cond.view(cond.size(0), -1).mean(dim=1, keepdim=True)
    return raw_energy# + cond

def scale_level(t):
  dd = 32 * 32 * 3
  sigma = 1e-3 + t * (5.0 - 1e-3)
  eps = 0.001 * sigma
  sigma2 = sigma ** 2
  eps2 = eps ** 2
  sigma4 = sigma2 ** 2
  factor = 4 * sigma4 / ((3 * dd * eps2 + dd * (dd - 1) * eps2 + 4 * dd * sigma2) * eps2)
  #factor = sigma4 / (dd * eps2)
  #factor = sigma2 / (dd * eps2)
  return sigma2# * torch.ones_like(factor)

def grad_action(params):
  with torch.no_grad():
    bad = False
    for param in params:
      if torch.isnan(param.grad).any():
        bad = True
        break
    if bad:
      for param in params:
        param.grad.zero_()
    else:
      torch.nn.utils.clip_grad_norm_(params, 5.0)

if __name__ == "__main__":
  opt = parse_options(
    "CIFAR10 EBM using RSM in flex.",
    path="/g/korbel/mjendrusch/runs/experimental/cifar10-recovery-16-simple",
    device="cuda:0",
    batch_size=64,
    max_epochs=1000,
    report_interval=1000,
    checkpoint_interval=50000,
  )

  cifar10 = CIFAR10("examples/", download=False, transform=ToTensor())
  data = CIFAR10Dataset(cifar10)
  data = DataDistribution(
    data, batch_size=opt.batch_size,
    device=opt.device
  )

  energy = Energy().to(opt.device)#AdaptedUNetEnergy().to(opt.device)

  training = recovery_likelihood_training(
    energy, data,
    optimizer=torch.optim.Adam,
    optimizer_kwargs=dict(lr=1e-4, betas=(0.9, 0.99)),
    level_weight=1.0,
    level_distribution=NormalNoise(lambda t: 1e-3 + t * (5.0 - 1e-3)),
    noise_distribution=RecoveryNoise(energy, lambda t: 1e-3 * torch.ones_like(t), scale=1.0),
    loss_scale=1.0 / 3072,
    ema_weight=0.9999,
    path=opt.path,
    device=opt.device,
    batch_size=opt.batch_size,
    max_epochs=opt.max_epochs,
    report_interval=opt.report_interval,
    checkpoint_interval=opt.checkpoint_interval
  )
  training.get_step("recovery_likelihood_step").extend_update(
    gradient_action=grad_action
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
