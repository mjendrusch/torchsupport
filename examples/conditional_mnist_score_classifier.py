import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from torchsupport.data.io import make_differentiable
from torchsupport.modules.basic import MLP
from torchsupport.modules.residual import ResNetBlock2d
from torchsupport.modules.normalization import AdaptiveInstanceNormPP
from torchsupport.training.samplers import Langevin
from torchsupport.training.score_supervised import ScoreSupervisedTraining

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

class EnergyDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __getitem__(self, index):
    data, label_index = self.data[index]
    data = 0.99 * data + 0.01 * torch.rand_like(data)
    return data, label_index

  def __len__(self):
    return len(self.data)

class Classifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.process = MLP(28 * 28, 64, 128, depth=3, normalization=spectral_norm, batch_norm=False)
    self.predict = MLP(64, 10, 128, depth=3, normalization=spectral_norm)
    self.condition = MLP(1, 128, 128, depth=3, normalization=spectral_norm)
    self.scale = nn.Linear(128, 64)
    self.bias = nn.Linear(128, 64)

  def forward(self, inputs, sigma, *args):
    with torch.enable_grad():
      make_differentiable(inputs)
      processed = self.process(inputs.view(inputs.size(0), -1))
      condition = self.condition(sigma.view(sigma.size(0), -1))
      scale = self.scale(condition)
      bias = self.bias(condition)
      logits = self.predict(processed * scale + bias)
      energy = -logits.logsumexp(dim=1)
      score = torch.autograd.grad(
        energy, inputs, grad_outputs=torch.ones_like(energy),
        retain_graph=True, create_graph=True
      )[0]
    return score, logits

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
    self.predict = nn.Linear(16 * 2 ** depth, 10)

  def forward(self, inputs, noise):
    out = self.input(inputs)
    cond = torch.zeros(
      inputs.size(0), 10,
      dtype=inputs.dtype,
      device=inputs.device
    )
    offset = (torch.log(noise) / torch.log(torch.tensor(0.60))).long()
    cond[torch.arange(inputs.size(0)), offset.view(-1)] = 1
    connections = []
    for norm, block in zip(self.down_norm, self.down):
      out = func.elu(block(norm(out, cond)))
      connections.append(out)
    features = func.adaptive_avg_pool2d(out, 1)
    logits = self.predict(features.view(features.size(0), -1))
    for norm, block, shortcut in zip(self.up_norm, self.up, reversed(connections)):
      out = func.elu(block(norm(torch.cat((out, shortcut), dim=1), cond)))
    del connections
    return self.output(out), logits

def _next(idx):
  return 32# * 2 ** (idx // 3)

def upscale(size):
  def _inner(data):
    current_size = data.size(1)
    difference = size - data.size(1)
    if difference > 0:
      data = torch.cat((
        data,
        torch.zeros(
          data.shape[0], difference, *data.shape[2:],
          device=data.device, dtype=data.dtype
        )
      ), dim=1)
    return data
  return _inner

class ConvEnergy(nn.Module):
  def __init__(self, depth=4):
    super(ConvEnergy, self).__init__()
    self.preprocess = nn.Conv2d(1, _next(0), 1)
    self.blocks = nn.ModuleList([
      (nn.Conv2d(_next(idx), _next(idx + 1), 3, padding=1))
      for idx in range(depth)
    ])
    self.project = [
      upscale(_next(idx + 1))
      for idx in range(depth)
    ]
    self.bn = nn.ModuleList([
      AdaptiveInstanceNormPP(_next(idx + 1), 10)
      for idx in range(depth)
    ])
    self.postprocess = (nn.Conv2d(_next(depth), 128, 1))
    self.predict = (nn.Linear(128, 10))

  def forward(self, inputs, noise, *args):
    with torch.enable_grad():
      make_differentiable(inputs)

      cond = torch.zeros(
        inputs.size(0), 10,
        dtype=inputs.dtype,
        device=inputs.device
      )
      offset = (torch.log(noise) / torch.log(torch.tensor(0.60))).long()
      cond[torch.arange(inputs.size(0)), offset.view(-1)] = 1
      out = self.preprocess(inputs)
      count = 0
      for bn, proj, block in zip(self.bn, self.project, self.blocks):
        out = func.elu(bn(proj(out) + block(out), cond))
        count += 1
        if count % 5 == 0:
          out = func.avg_pool2d(out, 2)
      out = self.postprocess(out)
      out = func.adaptive_avg_pool2d(out, 1).view(-1, 128)
      logits = self.predict(out)
      energy = -logits.logsumexp(dim=1)
      score = -torch.autograd.grad(
        energy, inputs, torch.ones_like(energy),
        create_graph=True, retain_graph=True
      )[0]
      return score, logits

class MNISTEnergyTraining(ScoreSupervisedTraining):
  def prepare(self):
    data = torch.rand(1, 28, 28)
    return (data,)

  def data_key(self, data):
    if isinstance(data, (list, tuple)):
      return data
    else:
      return (data,)

  def each_generate(self, data, *args):
    samples = [torch.clamp(sample, 0, 1) for sample in data[0:10]]
    samples = torch.cat(samples, dim=-1)
    self.writer.add_image("samples", samples, self.step_id)

# class MNISTEnergyTraining(EnergySupervisedTraining):
#   def prepare(self):
#     data = torch.rand(1, 28, 28)
#     labels = torch.randint(10, (1,))[0]
#     return data, labels

#   def data_key(self, data):
#     data, *args = data
#     return (data, *args)

#   def each_generate(self, data, labels):
#     samples = [torch.clamp(sample, 0, 1) for sample in data[0:10]]
#     samples = torch.cat(samples, dim=-1)
#     self.writer.add_image("samples", samples, self.step_id)

if __name__ == "__main__":
  mnist = MNIST("examples/", download=False, transform=ToTensor())
  data = EnergyDataset(mnist)

  energy = UNetEnergy()

  training = MNISTEnergyTraining(
    energy, data,
    network_name="classifier-mnist-score/unet",
    device="cuda:0",
    batch_size=32,
    max_epochs=1000,
    report_interval=1000,
    verbose=True
  )

  training.train()
