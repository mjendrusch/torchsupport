import torch
import torch.nn as nn
import torch.nn.functional as func

class DistanceDiscriminator(nn.Module):
  def __init__(self, batch_size, out_size):
    super().__init__()
    self.batch_size = batch_size
    self.batch_combine = nn.Linear(batch_size, out_size)

  def forward(self, data):
    data = data.view(data.size(0), -1)
    dist = ((data[None, :] - data[:, None]).norm(dim=1) + 1e-6).log()
    return self.batch_combine(dist)

class DynamicAugmentation(nn.Module):
  def __init__(self, transforms, target=0.6,
               p=0.0, step=0.01, every=4):
    self.transforms = transforms
    self.target = target
    self.p = p
    self.step = step
    self.every = every
    self.tick = 0

  def update(self, result):
    if self.tick % self.every == 0:
      with torch.no_grad():
        sign = result.sign().mean()
        if sign < self.target:
          self.p += self.step
        else:
          self.p -= self.step
        self.p = max(0, min(1, self.p))
      self.tick = 0
    self.tick += 1

  def forward(self, data):
    for transform in self.transforms:
      aug = transform(data)
      mask = (torch.rand(aug.size(0)) < self.p).to(data.device)
      data = (~mask).float() * data + mask.float() * aug
    return data
