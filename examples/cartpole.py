from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.interacting.awr import AWRTraining
from torchsupport.interacting.awac import AWACTraining
from torchsupport.interacting.bdpi import BDPITraining
from torchsupport.interacting.shared_data import SharedModule
from torchsupport.interacting.policies.basic import RandomPolicy, CategoricalGreedyPolicy, CategoricalPolicy, EpsilonGreedyPolicy
from torchsupport.interacting.environments.cartpole import CartPole

from torchsupport.modules.basic import MLP

class Policy(MLP):
  data_type = namedtuple("Data", ["logits", "outputs"])
  def __init__(self, hidden_size=128, depth=3, value=False):
    super().__init__(
      4, 2,
      hidden_size=hidden_size,
      depth=depth,
      batch_norm=False
    )
    self.value = value

  def schema(self):
    return self.data_type(
      logits=torch.zeros(2, dtype=torch.float),
      outputs=None
    )

  def forward(self, inputs, **kwargs):
    result = super().forward(inputs)
    if self.value:
      result = 1 * result.sigmoid()
    return result

mode = sys.argv[1]
index = sys.argv[2]
training = ...
if mode == "bdpi":
  policy = Policy()
  value = MLP(4, 2, batch_norm=False, depth=3)
  agent = CategoricalPolicy(SharedModule(policy))
  env = CartPole()

  training = BDPITraining(
    policy, value, agent, env,
    network_name=f"awr-test/bdpi-cartpole-{index}",
    discount=0.99,
    clones=4,
    critic_updates=4,
    gradient_updates=20,
    batch_size=1024,
    buffer_size=100_000,
    device="cuda:0",
    verbose=True
  )

elif mode == "awr":
  policy = Policy()
  value = MLP(4, 1, batch_norm=False, depth=3)
  agent = CategoricalPolicy(SharedModule(policy))
  env = CartPole()
  training = AWRTraining(
    policy, value, agent, env,
    network_name=f"awr-test/awr-cartpole-{index}",
    verbose=True, beta=0.05,
    auxiliary_steps=1,
    discount=0.990,
    clip=20,
    device="cuda:0",
    batch_size=1024,
    policy_steps=5
  )

elif mode == "awac":
  policy = Policy(hidden_size=64, depth=2)
  value = Policy(hidden_size=64, depth=2)
  agent = CategoricalPolicy(SharedModule(policy))
  env = CartPole(scale=False)
  training = AWACTraining(
    policy, value, agent, env,
    network_name=f"awr-test/awac-cartpole-{index}",
    verbose=True, beta=0.05,
    auxiliary_steps=5,
    discount=0.99,
    clip=20,
    device="cuda:0",
    batch_size=1024,
    policy_steps=5
  )

training.train()
