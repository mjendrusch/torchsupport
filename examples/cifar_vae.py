import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset
from torch.distributions import Normal, Categorical, Distribution, Bernoulli, OneHotCategorical

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from torchsupport.distributions import StandardNormal, DistributionList
from torchsupport.distributions.mixture_of_logits import DiscretizedMixtureLogits
from torchsupport.data.match import MatchTensor, match_l2, match_l1, match_bce
from torchsupport.training.vae import VAETraining, Prior
from torchsupport.modules.rezero import ReZero

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

def match_this(x, y):
  return match_l2(255 * x, 255 * y)

class VAEDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __getitem__(self, index):
    data, label = self.data[index]
    return (data,)

  def __len__(self):
    return len(self.data)

class ResBlock(nn.Module):
  def __init__(self, in_size, out_size, kernel_size, depth=1):
    super().__init__()
    self.project_in = nn.Conv2d(in_size, in_size // 4, 1, bias=False)
    self.project_out = nn.Conv2d(in_size // 4, out_size, 1, bias=False)
    self.blocks = nn.ModuleList([
      nn.Conv2d(in_size // 4, in_size // 4, kernel_size, padding=kernel_size // 2)
      for idx in range(depth)
    ])
    self.zero = ReZero(out_size, initial_value=0.1)

  def forward(self, inputs):
    out = self.project_in(inputs)
    for block in self.blocks:
      out = func.gelu(block(out))
    return self.zero(inputs, self.project_out(out))

class TopDown(nn.Module):
  def __init__(self, depth=4, level_repeat=2, base=32):
    super().__init__()
    self.blocks = nn.ModuleList([
      ResBlock(base, base, 3, depth=2)
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
    nn.GELU(),
    nn.Conv2d(in_size, in_size, 3, padding=1),
    nn.GELU(),
    nn.Conv2d(in_size, in_size, 3, padding=1),
    nn.GELU(),
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
      ResBlock(base, base, 3, depth=2)
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
      item = func.dropout(item, 0.7)
      features = torch.cat((out, item), dim=1)
      mf = 1
      lf = 1
      dist = Normal(mean(features) * mf, (logvar(features) * lf).exp())
      dists.append(dist)
      sample = dist.rsample()
      out = block(out + 0.1 * mod(sample))
      if (idx + 1) % self.level_repeat == 0:
        out = func.interpolate(out, scale_factor=2)
    #   if (idx + 1) % self.level_repeat == 0 and idx < len(self.blocks) - 1:
    #     out = func.interpolate(out, scale_factor=2)
    # results.append(out)

    return dists, (results, out)

class GMM(Distribution):
  def __init__(self, logits, locs, scales):
    self.logits = logits.clone()
    self.locs = locs.clone()
    self.scales = scales.clone()

  def log_prob(self, x):
    normal = Normal(self.locs, self.scales)
    prob = self.logits.log_softmax(dim=1)[:, :, None]
    print(x.shape, self.locs.shape, self.scales.shape)
    normal_logprob = normal.log_prob(x[:, None])
    # normal_logprob = normal_logprob.view(*normal_logprob.shape[:2], -1)
    # prob = prob.view(*prob.shape[:2], -1)
    print(normal_logprob.shape, prob.shape)
    total = (prob + normal_logprob).logsumexp(dim=1)
    return total

  def rsample(self, sample_shape=torch.Size()):
    if len(sample_shape) > 0:
      cat = OneHotCategorical(logits=self.logits).sample(sample_shape=sample_shape)
    else:
      cat = OneHotCategorical(logits=self.logits.transpose(1, -1))
      cat = cat.sample(sample_shape=sample_shape).transpose(1, -1) 
    cat = cat[:, :, None]
    print("LS", self.locs.shape, cat.shape, self.logits.shape)
    loc = (self.locs * cat).sum(dim=1)
    scale = (self.scales * cat).sum(dim=1)
    dist = Normal(loc, scale)
    return dist.sample()

  def sample(self, sample_shape=torch.Size()):
    with torch.no_grad():
      return self.rsample(sample_shape=sample_shape)

class DeepGMMPrior(nn.Module):
  def __init__(self, depth=4, level_repeat=2, base=32, mixture=20, z=32):
    super().__init__()
    self.first_logits = nn.Parameter(torch.randn(1, mixture, requires_grad=True))
    self.first_mean = nn.Parameter(torch.randn(1, mixture, z, requires_grad=True))
    self.first_logvar = nn.Parameter(torch.zeros(1, mixture, z, requires_grad=True))
    self.position = nn.Conv2d(2 * base, base, 1)
    self.mean = nn.ModuleList([
      z_project(2 * base, mixture * z)
      for idx in range(depth * level_repeat)
    ])
    self.mean_factor = nn.ParameterList([
      nn.Parameter(torch.zeros(1, z, 1, 1, requires_grad=True))
      for idx in range(depth * level_repeat)
    ])
    self.logvar = nn.ModuleList([
      z_project(2 * base, mixture * z)
      for idx in range(depth * level_repeat)
    ])
    self.logvar_factor = nn.ParameterList([
      nn.Parameter(torch.zeros(1, z, 1, 1, requires_grad=True))
      for idx in range(depth * level_repeat)
    ])
    self.mix = nn.ModuleList([
      z_project(2 * base, mixture)
      for idx in range(depth * level_repeat)
    ])

    self.z = z
    self.level_repeat = level_repeat

  def position_embedding(self, data):
    size = data.size(2)
    x = torch.arange(size, dtype=torch.float, device=data.device)
    features = torch.cat([
      (x / 1000 ** (2 * idx / base))[None]
      for idx in range(base // 2)
    ], dim=0)
    features = torch.cat((features.sin(), features.cos()), dim=0)
    x = features[:, :, None].expand(features.size(0), size, size)
    y = features[:, None, :].expand(features.size(0), size, size)
    pos = torch.cat((x, y), dim=0)[None]
    pos = self.position(pos)
    return pos.expand_as(data)

  def forward(self, hidden):
    hidden, _ = hidden
    first_dist = GMM(
      self.first_logits,
      self.first_mean,
      self.first_logvar.exp()
    )
    dists = [first_dist]
    for h, mean, logvar, mix, mf, lf in zip(
      hidden[:-1], self.mean, self.logvar,
      self.mix, self.mean_factor, self.logvar_factor
    ):
      mf = 1
      lf = 1
      pos = self.position_embedding(h)
      h = torch.cat((h, pos), dim=1)
      means = mean(h).view(h.size(0), -1, self.z, *h.shape[2:])
      logvars = logvar(h).view(h.size(0), -1, self.z, *h.shape[2:])
      logits = mix(h)
      dist = GMM(logits, means, logvars.exp())
      dists.append(dist)
    return DistributionList(dists)

class GMMGenerator(nn.Module):
  def __init__(self, prior, bottom_up, decoder, scale=4, temp=1.0):
    super().__init__()
    self.prior = prior
    self.bottom_up = bottom_up
    self.decoder = decoder
    self.scale = scale
    self.temp = temp

  def forward(self, batch_size, temp=1.0):
    first_dist = GMM(
      self.prior.first_logits[0],
      self.prior.first_mean,
      self.prior.first_logvar.exp()
    )
    results = []
    sample = temp * first_dist.sample(sample_shape=batch_size)
    out = self.bottom_up.first(sample)
    print(out.shape, sample.shape)
    out = out.view(out.size(0), out.size(1), 1, 1)
    out = func.interpolate(out, scale_factor=self.scale)
    for idx, (block, mean, logvar, mix, mf, lf, mod, zero) in enumerate(zip(
      self.bottom_up.blocks,
      self.prior.mean, self.prior.logvar, self.prior.mix,
      self.prior.mean_factor, self.prior.logvar_factor,
      self.bottom_up.modifiers, self.bottom_up.zeros
    )):
      mf = 1
      lf = 1
      results.append(out)
      pos = self.prior.position_embedding(out)
      dpos = torch.cat((out, pos), dim=1)
      means = mean(dpos).view(dpos.size(0), -1, self.prior.z, *dpos.shape[2:])
      logvars = logvar(dpos).view(dpos.size(0), -1, self.prior.z, *dpos.shape[2:])
      logits = mix(dpos)
      dist = GMM(logits, means, logvars.exp())
      #dist = Normal(mean(dpos) * mf, (logvar(dpos) * lf).exp())
      print("F", means.shape, logvars.shape, logits.shape)
      sample = temp * dist.rsample()
      out = block(out + 0.1 * mod(sample))
      if (idx + 1) % self.bottom_up.level_repeat == 0 and idx < len(self.bottom_up.blocks) - 1:
        out = func.interpolate(out, scale_factor=2)
    return self.decoder.block(results[-1]).clamp(0, 1)

class DeepPrior(nn.Module):
  def __init__(self, depth=4, level_repeat=2, base=32, z=32):
    super().__init__()
    self.first_mean = nn.Parameter(torch.zeros(1, z, requires_grad=True))
    self.first_logvar = nn.Parameter(torch.zeros(1, z, requires_grad=True))
    self.position = nn.Conv2d(2 * base, base, 1)
    self.mean = nn.ModuleList([
      z_project(2 * base, z)
      for idx in range(depth * level_repeat)
    ])
    self.mean_factor = nn.ParameterList([
      nn.Parameter(torch.zeros(1, z, 1, 1, requires_grad=True))
      for idx in range(depth * level_repeat)
    ])
    self.logvar = nn.ModuleList([
      z_project(2 * base, z)
      for idx in range(depth * level_repeat)
    ])
    self.logvar_factor = nn.ParameterList([
      nn.Parameter(torch.zeros(1, z, 1, 1, requires_grad=True))
      for idx in range(depth * level_repeat)
    ])

    self.level_repeat = level_repeat

  def position_embedding(self, data):
    size = data.size(2)
    x = torch.arange(size, dtype=torch.float, device=data.device)
    features = torch.cat([
      (x / 1000 ** (2 * idx / base))[None]
      for idx in range(base // 2)
    ], dim=0)
    features = torch.cat((features.sin(), features.cos()), dim=0)
    x = features[:, :, None].expand(features.size(0), size, size)
    y = features[:, None, :].expand(features.size(0), size, size)
    pos = torch.cat((x, y), dim=0)[None]
    pos = self.position(pos)
    return pos.expand_as(data)

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
      mf = 1
      lf = 1
      pos = self.position_embedding(h)
      h = torch.cat((h, pos), dim=1)
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
      mf = 1
      lf = 1
      results.append(out)
      pos = self.prior.position_embedding(out)
      dpos = torch.cat((out, pos), dim=1)
      dist = Normal(mean(dpos) * mf, (logvar(dpos) * lf).exp())
      sample = temp * dist.rsample()
      out = block(out + 0.1 * mod(sample))
      if (idx + 1) % self.bottom_up.level_repeat == 0 and idx < len(self.bottom_up.blocks) - 1:
        out = func.interpolate(out, scale_factor=2)
    res = DiscretizedMixtureLogits(10, self.decoder.block(results[-1])).sample()
    res = ((res + 1) / 2).clamp(0, 1)
    return res
    # return self.decoder.block(results[-1]).clamp(0, 1)

class DeepEncoder(nn.Module):
  def __init__(self, depth=4, level_repeat=2, level_repeat_down=None, base=32, z=32, scale=4):
    super().__init__()
    level_repeat_down = level_repeat_down or level_repeat
    self.project = nn.Conv2d(3, base, 3, padding=1)
    self.top_down = TopDown(
      depth=depth, base=base,
      level_repeat=level_repeat_down
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
      nn.GELU(),
      nn.Conv2d(base, base, 3, padding=1),
      nn.GELU(),
      nn.Conv2d(base, 3, 3, padding=1)
    )
    self.logvar = nn.Parameter(torch.zeros(1, 3, 1, 1, requires_grad=True))

  def display(self, output):
    return output.loc.clamp(0, 1)

  def forward(self, dists, other):
    results, out = other
    return Normal(self.block(results[-1]), (-3 + func.softplus(self.logvar + 3)).exp())

class LogisticDeepDecoder(nn.Module):
  def __init__(self, base=32):
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(base, base, 3, padding=1),
      nn.GELU(),
      nn.Conv2d(base, base, 3, padding=1),
      nn.GELU(),
      nn.Conv2d(base, 10 * 10, 3, padding=1)
    )

  def display(self, output):
    out = (output.sample() + 1) / 2
    return out

  def forward(self, dists, other):
    results, out = other
    out = DiscretizedMixtureLogits(10, self.block(results[-1]))
    return out

class CIFARVAETraining(VAETraining):
  def generate_samples(self):
    with torch.no_grad():
      generator = Generator(self.prior, self.encoder.bottom_up, self.decoder)
      sample = generator((self.batch_size,))
      print(sample.shape)
      return sample

if __name__ == "__main__":
  # with torch.autograd.detect_anomaly():
    mnist = CIFAR10("examples/", download=True, transform=ToTensor())
    data = VAEDataset(mnist)

    # z = 8
    # base=8

    # encoder = DeepEncoder(z=z, level_repeat=4, base=base)
    # decoder = DeepDecoder(base=base)

    # training = CIFARVAETraining(
    #   encoder, decoder,
    #   DeepGMMPrior(level_repeat=4, base=base, z=z), data,
    #   network_name="mnist-vae-very-deep/cifar-cal-noproject-gmm-32",
    #   device="cuda:0",
    #   batch_size=8,
    #   max_epochs=1000,
    #   prior_mu=1.0,
    #   verbose=True,
    #   generate=True,
    #   reconstruction_weight=1 / (3 * 32 * 32),
    #   divergence_weight=1 / (3 * 32 * 32)
    # ).load()

    z = 16
    base = 384

    encoder = DeepEncoder(z=z, level_repeat=10, level_repeat_down=4, base=base)
    decoder = DeepDecoder(base=base)

    training = CIFARVAETraining(
      encoder, decoder,
      DeepPrior(level_repeat=10, base=base, z=z), data,
      network_name="mnist-vae-very-deep-4/cifar-normal-noproject-noscale-36",
      device="cuda:0",
      batch_size=4,
      max_epochs=1000,
      prior_mu=1.0,
      verbose=True,
      generate=True,
      reconstruction_weight=1 / (3 * 32 * 32),
      divergence_weight=1 / (3 * 32 * 32)
    ).load()

    training.train()

    # z = 8
    # base=8

    # encoder = DeepEncoder(z=z, level_repeat=10, base=base)
    # decoder = DeepDecoder(base=base)

    # training = CIFARVAETraining(
    #   encoder, decoder,
    #   DeepPrior(level_repeat=10, base=base, z=z), data,
    #   network_name="mnist-vae-very-deep/cifar-cal-noproject-noscale-31",
    #   device="cuda:0",
    #   batch_size=8,
    #   max_epochs=1000,
    #   prior_mu=1.0,
    #   verbose=True,
    #   generate=True,
    #   reconstruction_weight=1 / (3 * 32 * 32),
    #   divergence_weight=1 / (3 * 32 * 32)
    # ).load()

    # training.train()

    # z = 8
    # base=8

    # encoder = DeepEncoder(z=z, level_repeat=20, base=base)
    # decoder = DeepDecoder(base=base)

    # training = CIFARVAETraining(
    #   encoder, decoder,
    #   DeepPrior(level_repeat=20, base=base, z=z), data,
    #   network_name="mnist-vae-very-deep/cifar-cal-noproject-no0-deep-31",
    #   device="cuda:0",
    #   batch_size=8,
    #   max_epochs=1000,
    #   prior_mu=1.0,
    #   verbose=True,
    #   generate=True,
    #   reconstruction_weight=1 / (3 * 32 * 32),
    #   divergence_weight=1 / (3 * 32 * 32)
    # ).load()

    # training.train()
