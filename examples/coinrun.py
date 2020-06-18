from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.interacting.awr import AWRTraining
from torchsupport.interacting.awac import AWACTraining
from torchsupport.interacting.bdpi import BDPITraining
from torchsupport.interacting.shared_data import SharedModule
from torchsupport.interacting.policies.basic import RandomPolicy, CategoricalGreedyPolicy, CategoricalPolicy, EpsilonGreedyPolicy
from torchsupport.interacting.environments.coinrun import CoinRun

from torchsupport.modules.basic import MLP

class Policy(nn.Module):
  data_type = namedtuple("Data", ["logits", "outputs"])
  def __init__(self, in_size=3, out_size=15):
    super().__init__()
    self.blocks = nn.Sequential(
      nn.Conv2d(in_size, 32, 3),
      nn.ReLU(),
      nn.InstanceNorm2d(32),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 32, 3),
      nn.InstanceNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 32, 3),
      nn.InstanceNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 32, 3),
      nn.InstanceNorm2d(32),
      nn.ReLU()
    )
    self.postprocess = nn.Linear(32, out_size)

  def schema(self):
    return self.data_type(
      logits=torch.zeros(15, dtype=torch.float),
      outputs=None
    )

  def forward(self, inputs, **kwargs):
    result = self.blocks(inputs)
    result = func.adaptive_avg_pool2d(result, 1).view(result.size(0), -1)
    result = self.postprocess(result)
    return result

mode = sys.argv[1]
index = sys.argv[2]
training = ...
if mode == "bdpi":
  policy = Policy()
  value = MLP(4, 2, batch_norm=False, depth=3)
  agent = CategoricalPolicy(SharedModule(policy))
  env = CoinRun()

  training = BDPITraining(
    policy, value, agent, env,
    network_name=f"awr-test/bdpi-coinrun-{index}",
    discount=0.99,
    clones=4,
    critic_updates=4,
    gradient_updates=20,
    batch_size=128,
    buffer_size=100_000,
    device="cuda:0",
    verbose=True
  )

elif mode == "awr":
  policy = Policy(in_size=3)
  value = Policy(in_size=3, out_size=1)
  agent = CategoricalPolicy(SharedModule(policy))
  env = CoinRun(history=1)
  training = AWRTraining(
    policy, value, agent, env,
    network_name=f"awr-test/awr-coinrun-{index}",
    verbose=True, beta=0.05,
    auxiliary_steps=1,
    discount=0.990,
    clip=20,
    n_workers=8,
    device="cuda:0",
    batch_size=128,
    buffer_size=50_000,
    policy_steps=1
  )

elif mode == "awac":
  policy = Policy(in_size=3)
  value = Policy(in_size=3)
  agent = CategoricalPolicy(SharedModule(policy))
  env = CoinRun(history=1)
  training = AWACTraining(
    policy, value, agent, env,
    network_name=f"awr-test/awac-coinrun-{index}",
    verbose=True, beta=1.0,
    auxiliary_steps=5,
    discount=0.990,
    clip=20,
    n_workers=8,
    device="cuda:0",
    batch_size=128,
    buffer_size=100_000,
    policy_steps=5
  )

training.train()
