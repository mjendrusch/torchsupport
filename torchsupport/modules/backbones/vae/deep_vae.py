import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal

from torchsupport.modules import ReZero
from torchsupport.distributions import DistributionList

class ResBlock(nn.Module):
  def __init__(self, size, kernel_size, depth=1):
    super().__init__()
    self.blocks = nn.ModuleList([
      nn.Conv2d(size, size, kernel_size, padding=kernel_size // 2)
      for idx in range(depth)
    ])
    self.zero = ReZero(size)

  def forward(self, inputs):
    out = inputs
    for block in self.blocks:
      out = func.relu(block(out))
    return self.zero(inputs, out)

class TopDown(nn.Module):
  def __init__(self, depth=4, level_repeat=2, base=32):
    super().__init__()
    self.blocks = nn.ModuleList([
      ResBlock(base, 3, depth=2)
      for idx in range(depth * level_repeat)
    ])
    self.last = nn.Linear(base, base)
    self.level_repeat = level_repeat

  def forward(self, inputs):
    out = inputs
    results = []
    for idx, block in enumerate(self.blocks):
      out = block(out)
      if (idx + 1) % self.level_repeat == 0:
        results = [out] + results
        out = func.avg_pool2d(out, 2)
    out = func.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
    out = self.last(out)
    results = [out] + results
    return results

def z_project(in_size, out_size):
  return nn.Sequential(
    nn.Conv2d(in_size, in_size, 1),
    nn.ReLU(),
    nn.Conv2d(in_size, in_size, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(in_size, in_size, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(in_size, out_size, 3, padding=1)
  )

class BottomUp(nn.Module):
  def __init__(self, depth=4, level_repeat=2, scale=4, base=32, z=32):
    super().__init__()
    self.first = nn.Linear(z, base)
    self.first_mean = nn.Linear(base, z)
    self.first_mean_factor = nn.Parameter(torch.zeros(1, z, requires_grad=True))
    self.first_logvar = nn.Linear(base, z)
    self.first_logvar_factor = nn.Parameter(torch.zeros(1, z, requires_grad=True))
    self.blocks = nn.ModuleList([
      ResBlock(base, 3, depth=2)
      for idx in range(depth * level_repeat)
    ])
    self.modifiers = nn.ModuleList([
      nn.Conv2d(z, base, 1, bias=False)
      for idx in range(depth * level_repeat)
    ])
    self.zeros = nn.ModuleList([
      ReZero(base)
      for idx in range(depth * level_repeat)
    ])
    self.mean = nn.ModuleList([
      z_project(2 * base, z)
      for idx in range(depth * level_repeat)
    ])
    self.logvar = nn.ModuleList([
      z_project(2 * base, z)
      for idx in range(depth * level_repeat)
    ])
    self.mean_factor = nn.ParameterList([
      nn.Parameter(torch.zeros(1, z, 1, 1, requires_grad=True))
      for idx in range(depth * level_repeat)
    ])
    self.logvar_factor = nn.ParameterList([
      nn.Parameter(torch.zeros(1, z, 1, 1, requires_grad=True))
      for idx in range(depth * level_repeat)
    ])
    self.level_repeat = level_repeat
    self.scale = scale

  def forward(self, inputs):
    first_dist = Normal(
      self.first_mean(inputs[0]) * self.first_mean_factor,
      (self.first_logvar(inputs[0]) * self.first_logvar_factor).exp()
    )
    first_sample = first_dist.rsample()
    out = self.first(first_sample)
    dists = [first_dist]
    results = []
    inputs = [
      x
      for item in inputs[1:]
      for x in [item] * self.level_repeat
    ]
    out = out.view(out.size(0), out.size(1), 1, 1)
    out = func.interpolate(out, scale_factor=self.scale)
    for idx, (item, block, mean, logvar, mf, lf, mod, zero) in enumerate(zip(
      inputs, self.blocks,
      self.mean, self.logvar,
      self.mean_factor, self.logvar_factor,
      self.modifiers, self.zeros
    )):
      results.append(out)
      features = torch.cat((out, item), dim=1)
      dist = Normal(mean(features) * mf, (logvar(features) * lf).exp())
      dists.append(dist)
      sample = dist.rsample()
      out = block(zero(out, mod(sample)))
      if (idx + 1) % self.level_repeat == 0:
        out = func.interpolate(out, scale_factor=2)

    return dists, (results, out)

class DeepPrior(nn.Module):
  def __init__(self, depth=4, level_repeat=2, base=32, z=32):
    super().__init__()
    self.first_mean = nn.Parameter(torch.zeros(1, z, requires_grad=True))
    self.first_logvar = nn.Parameter(torch.zeros(1, z, requires_grad=True))
    self.mean = nn.ModuleList([
      z_project(base, z)
      for idx in range(depth * level_repeat)
    ])
    self.mean_factor = nn.ParameterList([
      nn.Parameter(torch.zeros(1, z, 1, 1, requires_grad=True))
      for idx in range(depth * level_repeat)
    ])
    self.logvar = nn.ModuleList([
      z_project(base, z)
      for idx in range(depth * level_repeat)
    ])
    self.logvar_factor = nn.ParameterList([
      nn.Parameter(torch.zeros(1, z, 1, 1, requires_grad=True))
      for idx in range(depth * level_repeat)
    ])

    self.level_repeat = level_repeat

  def forward(self, hidden):
    hidden, _ = hidden
    first_dist = Normal(
      self.first_mean,
      self.first_logvar.exp()
    )
    dists = [first_dist]
    for h, mean, logvar, mf, lf in zip(
      hidden[:-1], self.mean, self.logvar,
      self.mean_factor, self.logvar_factor
    ):
      dist = Normal(mean(h) * mf, (logvar(h) * lf).exp())
      dists.append(dist)
    return DistributionList(dists)

class Generator(nn.Module):
  def __init__(self, prior, bottom_up, decoder, scale=4, temp=1.0):
    super().__init__()
    self.prior = prior
    self.bottom_up = bottom_up
    self.decoder = decoder
    self.scale = scale
    self.temp = temp

  def forward(self, batch_size, temp=1.0):
    first_dist = Normal(
      self.prior.first_mean,
      self.prior.first_logvar.exp()
    )
    results = []
    sample = temp * first_dist.sample(sample_shape=batch_size)[:, 0, :]
    out = self.bottom_up.first(sample)
    out = out.view(out.size(0), out.size(1), 1, 1)
    out = func.interpolate(out, scale_factor=self.scale)
    for idx, (block, mean, logvar, mf, lf, mod, zero) in enumerate(zip(
      self.bottom_up.blocks,
      self.prior.mean, self.prior.logvar,
      self.prior.mean_factor, self.prior.logvar_factor,
      self.bottom_up.modifiers, self.bottom_up.zeros
    )):
      results.append(out)
      dist = Normal(mean(out) * mf, (logvar(out) * lf).exp())
      sample = temp * dist.rsample()
      out = block(zero(out, mod(sample)))
      if (idx + 1) % self.bottom_up.level_repeat == 0 and idx < len(self.bottom_up.blocks) - 1:
        out = func.interpolate(out, scale_factor=2)
    return self.decoder.block(results[-1]).clamp(0, 1)

class DeepEncoder(nn.Module):
  def __init__(self, depth=4, level_repeat=2, base=32, z=32, scale=4):
    super().__init__()
    self.project = nn.Conv2d(3, base, 3, padding=1)
    self.top_down = TopDown(
      depth=depth, base=base,
      level_repeat=level_repeat
    )
    self.bottom_up = BottomUp(
      depth=depth, level_repeat=level_repeat,
      scale=scale, base=base, z=z
    )

  def forward(self, inputs):
    results = self.top_down(self.project(inputs))
    dists, (results, out) = self.bottom_up(results)
    return DistributionList(dists), (results, out)

class DeepDecoder(nn.Module):
  def __init__(self, base=32):
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(base, base, 3, padding=1),
      nn.ReLU(),
      nn.Conv2d(base, base, 3, padding=1),
      nn.ReLU(),
      nn.Conv2d(base, 3, 3, padding=1)
    )
    self.logvar = nn.Parameter(torch.zeros(1, 3, 1, 1, requires_grad=True))

  def display(self, output):
    return output.loc.clamp(0, 1)

  def forward(self, dists, other):
    results, out = other
    return Normal(self.block(results[-1]), self.logvar.exp())
