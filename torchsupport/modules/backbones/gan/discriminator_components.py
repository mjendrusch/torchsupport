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
    super().__init__()
    self.transforms = transforms
    self.target = target
    self.p = p
    self.step = step
    self.every = every
    self.tick = 0

  def update(self, result):
    if self.tick % self.every == 0:
      with torch.no_grad():
        sign = (result.sign() + 1).mean() / 2
        if sign < self.target:
          self.p -= self.step
        else:
          self.p += self.step
        self.p = max(0, min(1, self.p))
      self.tick = 0
    self.tick += 1

  def forward(self, data):
    for transform in self.transforms:
      aug = transform(data)
      if isinstance(data, (list, tuple)):
        tmp = []
        mask = (torch.rand(aug[0].size(0)) < self.p).to(aug[0].device)
        for item, aug_item in zip(data, aug):
          mm = mask.view(mask.size(0), *([1] * (item.dim() - 1)))
          item = ((~mm).float() * item + mm.float() * aug_item)
          tmp.append(item)
        data = tuple(tmp)
      else:
        mask = (torch.rand(aug.size(0)) < self.p).to(data.device)
        mask = mask.view(mask.size(0), *([1] * (data.dim() - 1)))
        data = ((~mask).float() * data + mask.float() * aug)
    return data
