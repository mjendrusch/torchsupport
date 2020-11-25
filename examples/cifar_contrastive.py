import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import Dataset

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, RandomAffine, ColorJitter

from torchsupport.training.contrastive import SimSiamTraining
from torchsupport.modules import MLP
from torchsupport.modules.rezero import ReZero

class ContrastiveDataset(Dataset):
  def __init__(self, data, transform, variants=2):
    self.data = data
    self.transform = transform
    self.variants = variants

  def __getitem__(self, index):
    data, _ = self.data[index]
    variants = [
      self.transform(data)
      for idx in range(self.variants)
    ]
    return variants

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

class SimpleResNet(nn.Module):
  def __init__(self, features=128, depth=4, level_repeat=2, base=32):
    super().__init__()
    self.project = nn.Conv2d(3, base, 1)
    self.blocks = nn.ModuleList([
      ResBlock(base, base, 3, depth=1)
      for idx in range(depth * level_repeat)
    ])
    self.last = MLP(base, features, features)
    self.level_repeat = level_repeat

  def forward(self, inputs):
    out = self.project(inputs)
    for idx, block in enumerate(self.blocks):
      out = block(out)
      if (idx + 1) % self.level_repeat == 0:
        out = func.avg_pool2d(out, 2)
    out = func.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
    out = self.last(out)
    return out

if __name__ == "__main__":
  cifar = CIFAR10("examples/", download=True)
  data = ContrastiveDataset(cifar, Compose([
    ColorJitter(1.0, 1.0, 1.0, 0.5),
    RandomAffine(60, (0.5, 0.5), (0.5, 2.0), 60),
    ToTensor()
  ]), variants=2)

  base = 16
  features = 64
  net = SimpleResNet(features=features, base=base, level_repeat=4)
  predictor = MLP(features, features, hidden_size=32)

  training = SimSiamTraining(
    net, predictor, data,
    network_name="cifar-contrastive/siam-5",
    device="cuda:0",
    batch_size=32,
    max_epochs=1000,
    verbose=True
  ).load()

  training.train()
