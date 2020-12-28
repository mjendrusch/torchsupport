import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Distribution, Normal

from torchsupport.modules.rezero import ReZero
from torchsupport.data.namedtuple import NamedTuple

class ResidualProjection(nn.Module):
  def __init__(self, in_size, out_size, hidden_size=64,
               depth=3, activation=None, pool=False):
    super().__init__()
    activation = activation or nn.ReLU()
    self.project_in = nn.Conv2d(in_size, hidden_size, 1)
    self.project_out = nn.Conv2d(hidden_size, out_size, 1)
    if pool:
      self.project_out = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.Linear(hidden_size, out_size)
      )
    self.blocks = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
        activation,
        nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
        activation
      )
      for idx in range(depth)
    ])
    self.zeros = nn.ModuleList([
      ReZero(hidden_size)
      for idx in range(depth)
    ])

  def forward(self, inputs):
    out = self.project_in(inputs)
    for block, zero in zip(self.blocks, self.zeros):
      out = zero(out, block(out))
    return self.project_out(out)

class ScaleBlock(nn.Module):
  def __init__(self, backbone, prior, posterior, policy, task):
    super().__init__()
    self.spatial = backbone
    self.prior = prior
    self.posterior = posterior
    self.policy = policy
    self.task = task

  def forward(self, inputs, mask=None, sample=None):
    shape = inputs.shape
    ind = torch.arange(shape[0], dtype=torch.long, device=inputs.device)[:, None]
    out = inputs
    out = self.spatial(out)
    policy = self.policy(out)[:, 0]
    prior = self.prior(out)
    # TODO more flexible priors
    if isinstance(prior, Distribution):
      prior_sample = prior.sample()
    else:
      prior_sample = 1.0 * prior
      prior = Normal(prior, 1.0)
    if mask:
      prior_sample[ind, :, mask[0], mask[1]] = sample
      policy = policy[ind, mask[0], mask[1]]
      prior_loc = prior.loc[ind, :, mask[0], mask[1]]
      prior = Normal(prior_loc, 1.0)
    else:
      policy = None
    out = torch.cat((out, prior_sample), dim=1)
    sample = self.posterior(out)
    task = self.task(sample)
    return task, NamedTuple(prior=prior, posterior=sample, policy=policy)

class ScaleResNet(ScaleBlock):
  def __init__(self, backbone, feature_size=64,
               out_size=10, hidden_size=64,
               latent_size=64, prior_depth=3,
               posterior_depth=3, policy_depth=3,
               activation=None):
    super().__init__(
      backbone,
      prior=ResidualProjection(
        feature_size, latent_size,
        hidden_size=hidden_size,
        depth=prior_depth,
        activation=activation
      ),
      posterior=ResidualProjection(
        feature_size + latent_size, latent_size,
        hidden_size=hidden_size,
        depth=posterior_depth,
        activation=activation,
        pool=True
      ),
      policy=ResidualProjection(
        feature_size, 1,
        hidden_size=hidden_size,
        depth=policy_depth,
        activation=activation
      ),
      task=nn.Linear(latent_size, out_size)
    )
