import time

import torch
import torch.nn as nn
import torch.nn.functional as func

from tensorboardX import SummaryWriter

from torchsupport.data.io import netread, netwrite, to_device

from torchsupport.training.training import Training
from torchsupport.training.state import (
  NetState, NetNameListState, TrainingState
)

from torchsupport.interacting.buffer import SchemaBuffer
from torchsupport.interacting.collector_task import EnvironmentCollector
from torchsupport.interacting.distributor_task import DefaultDistributor
from torchsupport.interacting.data_collector import ExperienceCollector
from torchsupport.interacting.stats import ExperienceStatistics

class OffPolicyTraining(Training):
  checkpoint_parameters = Training.checkpoint_parameters + [
    TrainingState(),
    NetNameListState("auxiliary_names"),
    NetState("policy"),
    NetState("optimizer"),
    NetState("auxiliary_optimizer")
  ]
  def __init__(self, policy, agent, environment,
               auxiliary_networks=None,
               buffer_size=100_000,
               piecewise_append=False,
               policy_steps=1,
               auxiliary_steps=1,
               n_workers=8,
               discount=0.99,
               double=False,
               optimizer=torch.optim.Adam,
               optimizer_kwargs=None,
               aux_optimizer=torch.optim.Adam,
               aux_optimizer_kwargs=None,
               **kwargs):
    super().__init__(**kwargs)
    self.policy_steps = policy_steps
    self.auxiliary_steps = auxiliary_steps

    self.current_losses = {}

    self.statistics = ExperienceStatistics()
    self.discount = discount
    self.environment = environment
    self.agent = agent
    self.policy = policy.to(self.device)
    self.collector = EnvironmentCollector(environment, agent, discount=discount)
    self.distributor = DefaultDistributor()
    self.data_collector = ExperienceCollector(
      self.distributor, self.collector,
      n_workers=n_workers, piecewise=piecewise_append
    )
    self.buffer = SchemaBuffer(
      self.data_collector.schema(), buffer_size, double=double
    )

    optimizer_kwargs = optimizer_kwargs or {}
    self.optimizer = optimizer(
      self.policy.parameters(), **optimizer_kwargs
    )

    auxiliary_netlist = []
    self.auxiliary_names = []
    for network in auxiliary_networks:
      self.auxiliary_names.append(network)
      network_object = auxiliary_networks[network].to(self.device)
      setattr(self, network, network_object)
      auxiliary_netlist.extend(list(network_object.parameters()))

    aux_optimizer_kwargs = aux_optimizer_kwargs or {}
    self.auxiliary_optimizer = aux_optimizer(
      auxiliary_netlist, **aux_optimizer_kwargs
    )

    self.checkpoint_names = dict(
      policy=self.policy,
      **{
        name: getattr(self, name)
        for name in self.auxiliary_names
      }
    )

  def run_policy(self, sample):
    raise NotImplementedError("Abstract.")

  def policy_loss(self, *args):
    raise NotImplementedError("Abstract.")

  def run_auxiliary(self, sample):
    raise NotImplementedError("Abstract.")

  def auxiliary_loss(self, *args):
    raise NotImplementedError("Abstract.")

  def policy_step(self):
    self.optimizer.zero_grad()

    data = self.buffer.sample(self.batch_size)
    data = to_device(data, self.device)

    args = self.run_policy(data)
    loss = self.policy_loss(*args)
    loss.backward()
    self.optimizer.step()

    self.current_losses["policy"] = float(loss)

    self.agent.push()

  def auxiliary_step(self):
    self.auxiliary_optimizer.zero_grad()

    data = self.buffer.sample(self.batch_size)
    data = to_device(data, self.device)

    args = self.run_auxiliary(data)
    loss = self.auxiliary_loss(*args)
    loss.backward()

    self.current_losses["auxiliary"] = float(loss)

    self.auxiliary_optimizer.step()

  def step(self):
    self.current_losses["reward"] = float(self.statistics.total)
    self.current_losses["length"] = float(self.statistics.length)
    self.current_losses["environment steps"] = float(self.statistics.steps)

    for _ in range(self.auxiliary_steps):
      self.auxiliary_step()
    for _ in range(self.policy_steps):
      self.policy_step()

    if self.verbose:
      for loss_name in self.current_losses:
        loss_float = self.current_losses[loss_name]
        self.writer.add_scalar(f"{loss_name} loss", loss_float, self.step_id)

    self.each_step()

  def validate(self):
    pass # TODO

  def initialize(self):
    self.data_collector.start(self.statistics, self.buffer)
    while len(self.buffer) < 1:
      print("waiting for samples...")
      time.sleep(5)

  def finalize(self):
    self.data_collector.join()

  def train(self):
    self.initialize()
    for _ in range(self.max_steps):
      self.step()
      self.log()
      self.step_id += 1

    self.finalize()

    return self.policy
