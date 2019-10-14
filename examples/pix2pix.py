import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import Dataset

from torchvision.datasets import ImageFolder

from torchsupport.modules.basic import MLP
from torchsupport.training.gan import NormalizedDiversityGANTraining
from torchsupport.training.translation import CycleGANTraining, PairedGANTraining

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

  def forward(self, sample):
    sample, condition = sample
    source = condition.view(condition.size(0), -1)
    inputs = torch.cat((source, sample), dim=1)
    result = torch.sigmoid(self.generate(inputs))
    result = result.view(result.size(0), 3, 28, 28)
    return condition, result

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.preprocess = nn.Conv2d(6, 32, 1)
    self.blocks = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d((idx + 1) * 32, (idx + 1) * 32, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d((idx + 1) * 32),
        nn.Conv2d((idx + 1) * 32, (idx + 2) * 32, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d((idx + 2) * 32),
      )
      for idx in range(4)
    ])
    self.postprocess = nn.Linear(5 * 32, 1)

  def forward(self, data):
    condition, inputs = data
    combined = torch.cat((inputs, condition), dim=1)
    out = func.relu(self.preprocess(combined))
    for block in self.blocks:
      out = block(out)
      out = func.max_pool2d(out, 2)
    out = func.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
    return self.postprocess(out)

class E2SGANTraining(NormalizedDiversityGANTraining, PairedGANTraining):
  def __init__(self, *args, weight=1.0, alpha=0.8, **kwargs):
    PairedGANTraining.__init__(self, *args, **kwargs)
    NormalizedDiversityGANTraining.__init__(self, alpha=alpha, diversity_weight=weight)

  def each_generate(self, data, translated, sample):
    data_points = torch.cat([x for x in data[0][:5]], dim=2).detach()
    translated_points = torch.cat([x for x in translated[1][:5]], dim=2).detach()
    real_points = torch.cat([x for x in data[1][:5]], dim=2).detach()
    self.writer.add_image("data", data_points, self.step_id)
    self.writer.add_image("translated", translated_points, self.step_id)
    self.writer.add_image("real", real_points, self.step_id)

if __name__ == "__main__":
  data = MiniEdges2Shoes("~/Downloads/edges2shoes/")
  generators = Generator()
  discriminators = Discriminator()
  training = E2SGANTraining(
    generators, discriminators, data,
    network_name="pix2pix",
    device="cpu",
    batch_size=64,
    max_epochs=1000,
    n_critic=1,
    gamma=10,
    verbose=True
  )

  training.train()
