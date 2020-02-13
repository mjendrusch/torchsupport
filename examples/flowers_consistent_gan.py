import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from torchsupport.data.transforms import Compose, Crop
from torchsupport.modules.basic import MLP
from torchsupport.modules.residual import ResNetBlock2d
from torchsupport.modules.normalization import AdaptiveInstanceNormPP
from torchsupport.training.samplers import Langevin
from torchsupport.training.consistent_gan import ConsistentGANTraining

from matplotlib import pyplot as plt

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

class ConsistentDataset(Dataset):
  def __init__(self, data):
    self.data = data
    self.size = 64

  def levels(self, data):
    upper = data
    middle = upper[:, 64:-64, 64:-64]
    lower = middle[:, 32:-32, 32:-32]
    upper = func.adaptive_avg_pool2d(upper.unsqueeze(0), 64)[0]
    middle = func.adaptive_avg_pool2d(middle.unsqueeze(0), 64)[0]
    return (upper, middle, lower)

  def square_mask(self, size):
    _, w, h = size
    min_size = min(w, h)
    mask_size, = torch.randint(min_size // 2, 2 * min_size // 3, (1,))
    mask = torch.zeros(1, w, h)
    x, = torch.randint(0, w - mask_size + 1, (1,))
    y, = torch.randint(0, h - mask_size + 1, (1,))

    mask[:, x:x + mask_size, y:y + mask_size] = 1
    return mask

  def level_masks(self, levels):
    avl = []
    req = []
    for level in levels:
      requested = self.square_mask(level.size())
      available = 1 - requested
      req.append(requested)
      avl.append(available)
    return avl, req

  def __getitem__(self, index):
    data, label = self.data[index]
    levels = self.levels(data)
    available, requested = self.level_masks(levels)
    return levels, available, requested

  def __len__(self):
    return len(self.data)

class LevelGenerator(nn.Module):
  def __init__(self, size=64, z=32):
    super().__init__()
    self.z = z
    self.preprocess = spectral_norm(nn.Conv2d(5 + 3, size, 3, padding=1))
    self.noise = nn.Parameter(torch.rand(6, 1, size, 1, 1))
    self.post_noise = nn.Parameter(torch.rand(3, 1, size, 1, 1))
    self.bg = nn.Parameter(torch.randn(1, 64, 8, 8))
    self.color = nn.Conv2d(size, 3, 1)
    self.encoder = nn.ModuleList([
      spectral_norm(nn.Conv2d(size, size, 3, dilation=idx + 1, padding=idx + 1))
      for idx in range(6)
    ])
    self.encoder_norm = nn.ModuleList([
      nn.InstanceNorm2d(size)
      for idx in range(6)
    ])
    self.decoder = nn.ModuleList([
      nn.Conv2d(2 * size, size, 3, dilation=idx + 1, padding=idx + 1)
      for idx in reversed(range(6))
    ])
    self.decoder_norm = nn.ModuleList([
      AdaptiveInstanceNormPP(size, z)
      for idx in reversed(range(6))
    ])
    self.post = nn.ModuleList([
      nn.Conv2d(size, size, 3, dilation=1, padding=1)
      for idx in range(3)
    ])
    self.post_norm = nn.ModuleList([
      AdaptiveInstanceNormPP(size, z)
      for idx in reversed(range(3))
    ])

  def forward(self, sample, prior, restricted_inputs, available, requested):
    prior = func.interpolate(prior, scale_factor=2, mode='bilinear')
    inputs = torch.cat((prior, restricted_inputs, available, requested), dim=1)
    out = self.preprocess(inputs)
    skip = []
    for idx, (block, bn) in enumerate(zip(self.encoder, self.encoder_norm)):
      out = func.elu(block(bn(out))) + out
      skip.append(out)
      if (idx + 1) % 2 == 0:
        out = func.avg_pool2d(out, 2)

    rec = self.bg
    for ridx, (noise, block, bn) in enumerate(zip(self.noise, self.decoder, self.decoder_norm)):
      idx = 6 - ridx
      if idx % 2 == 0:
        rec = func.interpolate(rec, scale_factor=2, mode='bilinear')
      normed = bn(rec + noise * torch.randn_like(rec), sample)
      combined = torch.cat((normed, skip[idx - 1]), dim=1)
      rec = func.elu(block(combined))

    for idx, (noise, block, bn) in enumerate(zip(self.post_noise, self.post, self.post_norm)):
      normed = bn(rec + noise * torch.randn_like(rec), sample)
      rec = func.elu(block(normed)) + rec
    result = self.color(rec).sigmoid()

    return result

class Generator(nn.Module):
  def __init__(self, depth=3, size=64, z=1024):
    super().__init__()
    self.depth = depth
    self.size = size
    self.z = z
    self.generators = nn.ModuleList([
      LevelGenerator(self.size, self.z)
      for idx in range(self.depth)
    ])

  def sample(self, batch_size):
    return [
      torch.randn(batch_size, self.z)
      for idx in range(self.depth)
    ]

  def forward(self, sample, stages, restricted, available, requested):
    prior = torch.zeros_like(stages[0])
    stages = [prior] + stages
    generated = []
    new_stages = []
    for generator, sam, pri, res, avl, req in zip(
        self.generators, sample, stages,
        restricted, available, requested
    ):
      if pri is None:
        pri = prior
      gen = generator(sam, pri, res, avl, req)
      generated.append(gen)
      off = self.size // 2
      prior = gen[:, :, off:-off, off:-off]
      new_stages.append(pri)
    new_stages = new_stages[1:]
    return generated, new_stages

class LevelDiscriminator(nn.Module):
  def __init__(self, size=64):
    super().__init__()
    self.preprocess = nn.Conv2d(8, size, 3, padding=1)
    self.encoder = nn.ModuleList([
      nn.Conv2d(size, size, 3, dilation=1, padding=1)
      for idx in range(8)
    ])
    self.encoder_norm = nn.ModuleList([
      nn.InstanceNorm2d(size)
      for idx in range(8)
    ])
    self.decide = nn.Linear(size, 1)

  def forward(self, available_input, requested_input, available, requested):
    inputs = torch.cat((available_input, requested_input, available, requested), dim=1)
    out = self.preprocess(inputs)
    for idx, (block, bn) in enumerate(zip(self.encoder, self.encoder_norm)):
      out = func.elu(block(bn(out))) + out
      if (idx + 1) % 2 == 0:
        out = func.max_pool2d(out, 2)

    out = func.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
    result = self.decide(out)

    return result

class Discriminator(nn.Module):
  def __init__(self, size=64, depth=3):
    super().__init__()
    self.blocks = nn.ModuleList([
      LevelDiscriminator(size=size)
      for _ in range(depth)
    ])

  def forward(self, avail, reqst, a, r):
    decisions = []
    for block, avl, req, am, rm in zip(self.blocks, avail, reqst, a, r):
      decision = block(avl, req, am, rm)
      decisions.append(decision)
    return torch.cat(decisions, dim=0)

class FlowersGANTraining(ConsistentGANTraining):
  def each_generate(self, inputs, generated, stages, available, requested):
    for idx, (inp, gen) in enumerate(zip(inputs, generated)):
      view = inp[:10].detach().to("cpu")
      images = torch.cat([
        (image - image.min()) / (image.max() - image.min()) for image in view
      ], dim=2)
      self.writer.add_image(f"source {idx}", images, self.step_id)
      view = gen[:10].detach().to("cpu")
      images = torch.cat([
        (image - image.min()) / (image.max() - image.min()) for image in view
      ], dim=2)
      self.writer.add_image(f"sample {idx}", images, self.step_id)
    for idx, stage in enumerate(stages):
      view = stage[:10].detach().to("cpu")
      images = torch.cat([
        (image - image.min()) / (image.max() - image.min()) for image in view
      ], dim=2)
      self.writer.add_image(f"stage {idx}", images, self.step_id)

class Pool:
  def __call__(self, x):
    result = func.adaptive_avg_pool2d(x, (256, 256))
    return result

if __name__ == "__main__":
  mnist = ImageFolder("examples/flowers/", transform=Compose([
    ToTensor(),
    Crop(400, 401, 400, 401),
    Pool()
  ]))
  data = ConsistentDataset(mnist)

  gen = Generator()
  disc = Discriminator()

  training = FlowersGANTraining(
    gen, disc, data,
    network_name="consistent-gan/stylish-5",
    levels=[32, 32, 32],
    device="cuda:0",
    gamma=0.01,
    batch_size=8,
    max_epochs=1000,
    verbose=True
  ).load()

  training.train()
