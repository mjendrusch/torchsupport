import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal
from torch.utils.data import Dataset

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, RandomAffine, ColorJitter

from torchsupport.training.multiscale_training import MultiscaleClassifierTraining, MultiscaleNet
from torchsupport.modules import MLP
from torchsupport.modules.rezero import ReZero
from torchsupport.data.namedtuple import NamedTuple

class CIFARMultiscaleDataset(Dataset):
  def __init__(self, data, k=4, separate=False):
    self.data = data
    self.k = k
    self.separate = separate
    if self.separate:
      self.k = 1

  def __getitem__(self, index):
    data, label = self.data[index]
    low = func.interpolate(data[None], scale_factor=1 / 4, mode="bilinear")
    high = torch.cat(torch.cat(data[None].chunk(4, dim=3), dim=0)[None].chunk(4, dim=3), dim=0)
    perm = torch.randperm(4 * 4)
    x = (perm // 4)[:self.k]
    y = (perm % 4)[:self.k]
    high = high[x, y]
    mask = (x, y)
    masks = [None, mask]
    inputs = [high, low]

    if self.separate:
      return list(zip(inputs, [label, label]))
    return (inputs, masks), (label, label)

  def __len__(self):
    return len(self.data)

class ScaleBlock(nn.Module):
  def __init__(self):
    super().__init__()
    self.spatial = MLP(8 * 8 * 3, 4 * 4 * 32, batch_norm=False)
    self.prior = MLP(4 * 4 * 32, 4 * 4 * 32, batch_norm=False)
    self.posterior = nn.Linear(4 * 4 * (32 + 16), 16)
    self.policy = nn.Linear(4 * 4 * 32, 4 * 4 * 1)
    self.task = nn.Linear(16, 10)

  def forward(self, inputs, mask=None, sample=None):
    shape = inputs.shape
    ind = torch.arange(shape[0], dtype=torch.long, device=inputs.device)[:, None]
    out = inputs.view(-1, 8 * 8 * 3)
    out = self.spatial(out)
    policy = self.policy(out)
    policy = policy.view(-1, 4, 4)
    mu, logvar = self.prior(out).view(-1, 4 * 4, 32).chunk(2, dim=-1)
    prior = Normal(mu, 1.0)
    prior_sample = 1.0 * mu#prior.sample()
    prior_sample = prior_sample.view(-1, 4, 4, 16)
    if mask:
      prior_sample[ind, mask[0], mask[1]] = sample
      res = mu.view(-1, 4, 4, 16)[ind, mask[0], mask[1]]
      logvar = logvar.view(-1, 4, 4, 16)[ind, mask[0], mask[1]]
      prior = Normal(res, 1.0)
      policy = policy[ind, mask[0], mask[1]]
    else:
      policy = None
    out = out.view(-1, 4, 4, 32)
    out = torch.cat((out, prior_sample), dim=-1)
    sample = self.posterior(out.view(-1, 4 * 4 * (32 + 16)))
    task = self.task(sample)
    return task, NamedTuple(prior=prior, posterior=sample, policy=policy)

if __name__ == "__main__":
  cifar = CIFAR10("examples/", transform=ToTensor(), download=True)
  path_data = CIFARMultiscaleDataset(cifar, k=4)
  separate_data = CIFARMultiscaleDataset(cifar, separate=True)

  net = MultiscaleNet([
    ScaleBlock(),
    ScaleBlock()
  ])

  training = MultiscaleClassifierTraining(
    net, separate_data,
    stack_data=path_data,
    path_data=path_data,
    network_name="cifar-multiscale/17",
    device="cuda:0",
    batch_size=32,
    max_epochs=1000,
    verbose=True,
    n_path=10
  )

  training.train()
