import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import Dataset

from torchvision.datasets import ImageFolder

from torchsupport.modules.basic import MLP
from torchsupport.training.translation import AugmentedCycleGANTraining

class MiniEdges2Shoes(Dataset):
  def __init__(self, path, mode=0):
    self.data = ImageFolder(path)
    self.indices = [
      idx
      for idx, sample in enumerate(self.data.samples)
      if sample[1] == mode
    ]

  def __getitem__(self, index):
    position = self.indices[index]
    img, _ = self.data[position]
    img = torch.tensor(np.array(img)).permute(2, 0, 1).to(torch.float) / 255
    edge = img[:, :, :256].unsqueeze(0)
    shoe = img[:, :, 256:].unsqueeze(0)

    edge = func.adaptive_max_pool2d(1 - edge, (28, 28))
    shoe = func.adaptive_avg_pool2d(shoe, (28, 28))

    return edge[0], shoe[0]

  def __len__(self):
    return len(self.indices)

class UnpairedEdge2Shoes(MiniEdges2Shoes):
  def __init__(self, path, mode=0):
    super().__init__(path, mode=mode)

  def __getitem__(self, index):
    edge_index = torch.randint(0, len(self) - 1, (1,))[0]
    shoe_index = torch.randint(0, len(self) - 1, (1,))[0]

    edge, _ = super().__getitem__(edge_index)
    _, shoe = super().__getitem__(shoe_index)

    return edge, shoe

class Generator(nn.Module):
  def __init__(self, z=32):
    super().__init__()
    self.z = z
    self.generate = MLP(3 * 28 * 28 + z, 3 * 28 * 28, depth=4)

  def sample(self, batch_size):
    return torch.randn(batch_size, self.z)

  def forward(self, sample, condition):
    condition = condition.view(condition.size(0), -1)
    inputs = torch.cat((condition, sample), dim=1)
    result = torch.sigmoid(self.generate(inputs))
    result = result.view(result.size(0), 3, 28, 28)
    return result

class Encoder(nn.Module):
  def __init__(self, z=32):
    super().__init__()
    self.z = z
    self.source = MLP(3 * 28 * 28, 64, depth=4)
    self.target = MLP(3 * 28 * 28, 64, depth=4)
    self.combine = MLP(128, self.z)

  def sample(self, batch_size):
    return torch.randn(batch_size, self.z)

  def forward(self, source, target):
    sc = self.source(source.view(source.size(0), -1))
    tg = self.target(target.view(target.size(0), -1))
    combined = torch.cat((sc, tg), dim=1)
    return self.combine(combined)

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.preprocess = nn.Conv2d(3, 32, 1)
    self.blocks = nn.ModuleList([
      nn.Conv2d(32, 32, 3, padding=1)
      for idx in range(4)
    ])
    self.postprocess = nn.Linear(32, 1)

  def forward(self, data):
    out = func.relu(self.preprocess(data))
    for block in self.blocks:
      out = func.relu(block(out))
      out = func.max_pool2d(out, 2)
    out = func.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
    return self.postprocess(out)

class LatentDiscriminator(nn.Module):
  def __init__(self, z=32):
    super().__init__()
    self.process = MLP(z, 1, depth=3)

  def forward(self, inputs):
    return self.process(inputs.view(inputs.size(0), -1))

class E2SGANTraining(AugmentedCycleGANTraining):
  def each_generate(self, data, translated, cycled, *crap):
    data_points = torch.cat([x for x in data[0][:5]], dim=2).detach()
    translated_points = torch.cat([x for x in translated[0][:5]], dim=2).detach()
    cycled_points = torch.cat([x for x in cycled[0][:5]], dim=2).detach()
    self.writer.add_image("data", data_points, self.step_id)
    self.writer.add_image("translated", translated_points, self.step_id)
    self.writer.add_image("cycled", cycled_points, self.step_id)

if __name__ == "__main__":
  data = UnpairedEdge2Shoes("~/Downloads/edges2shoes/")
  generators = (Generator(), Generator())
  discriminators = (
    Discriminator(), Discriminator(),
    LatentDiscriminator(), LatentDiscriminator()
  )
  encoders = (Encoder(), Encoder())
  training = E2SGANTraining(
    generators, discriminators, encoders, data,
    network_name="e2s-aug-cycle",
    device="cpu",
    batch_size=64,
    max_epochs=1000,
    verbose=True
  )

  training.train()
