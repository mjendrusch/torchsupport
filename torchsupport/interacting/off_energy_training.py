import time

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import Dataset

from tensorboardX import SummaryWriter

from torchsupport.data.io import netread, netwrite, to_device

from torchsupport.training.training import Training
from torchsupport.training.state import (
  NetState, NetNameListState, TrainingState
)

from torchsupport.data.collate import DataLoader

from torchsupport.interacting.buffer import SchemaBuffer
from torchsupport.interacting.collector_task import EnergyCollector
from torchsupport.interacting.distributor_task import DefaultDistributor
from torchsupport.interacting.data_collector import ExperienceCollector
from torchsupport.interacting.stats import EnergyStatistics

class InfiniteWrapper(Dataset):
  def __init__(self, data, steps=1, batch=1):
    self.data = data
    self.steps = steps
    self.batch = batch

  def __getitem__(self, index):
    index = index % len(self.data)
    return self.data[index]

  def __len__(self):
    return self.steps * self.batch

class OffEnergyTraining(Training):
  checkpoint_parameters = Training.checkpoint_parameters + [
    TrainingState(),
    NetNameListState("auxiliary_names"),
    NetState("score"),
    NetState("optimizer"),
    NetState("auxiliary_optimizer")
  ]
  def __init__(self, score, energy, data,
               auxiliary_networks=None,
               integrator=None,
               decay=1.0,
               buffer_size=100_000,
               double=False,
               score_steps=1,
               auxiliary_steps=1,
               n_workers=8,
               optimizer=torch.optim.Adam,
               optimizer_kwargs=None,
               aux_optimizer=torch.optim.Adam,
               aux_optimizer_kwargs=None):
    self.score_steps = score_steps
    self.auxiliary_steps = auxiliary_steps

    self.current_losses = {}

    self.statistics = EnergyStatistics()
    self.integrator = integrator
    self.decay = decay
    self.score = score.to(self.device)
    self.energy = energy
    self.collector = EnergyCollector(
      energy, integrator, batch_size=2 * self.batch_size
    )
    self.distributor = DefaultDistributor()
    self.data_collector = ExperienceCollector(
      self.distributor, self.collector, n_workers=n_workers
    )
    self.buffer = SchemaBuffer(
      self.data_collector.schema(), buffer_size,
      double=double
    )

    self.data = data
    self.wrapped_data = InfiniteWrapper(
      data, steps=self.max_steps, batch=self.batch_size
    )
    self.data_loader = DataLoader(
      self.wrapped_data, batch_size=self.batch_size, num_workers=8,
      shuffle=True, drop_last=True
    )
    self.data_iter = iter(self.data_loader)

    optimizer_kwargs = optimizer_kwargs or {}
    self.optimizer = optimizer(
      self.score.parameters(), **optimizer_kwargs
    )

    auxiliary_netlist = []
    self.auxiliary_names = []
    if not auxiliary_networks:
      self.auxiliary_steps = 0

    auxiliary_networks = auxiliary_networks or {"_dummy": nn.Linear(1, 1)}
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
      score=self.score,
      **{
        name: getattr(self, name)
        for name in self.auxiliary_names
        if name != "_dummy"
      }
    )

  def run_score(self, sample, data):
    raise NotImplementedError("Abstract.")

  def score_loss(self, *args):
    raise NotImplementedError("Abstract.")

  def run_auxiliary(self, sample, data):
    raise NotImplementedError("Abstract.")

  def auxiliary_loss(self, *args):
    raise NotImplementedError("Abstract.")

  def score_step(self, data):
    self.optimizer.zero_grad()

    fake = self.buffer.sample(self.batch_size)
    fake = to_device(fake, self.device)
    data = to_device(data, self.device)

    args = self.run_score(fake, data)
    loss = self.score_loss(*args)
    loss.backward()
    self.optimizer.step()

    self.current_losses["ebm"] = float(loss)

    self.energy.push()

  def auxiliary_step(self, data):
    self.auxiliary_optimizer.zero_grad()

    fake = self.buffer.sample(self.batch_size)
    fake = to_device(fake, self.device)

    args = self.run_auxiliary(fake, data)
    loss = self.auxiliary_loss(*args)
    loss.backward()

    self.current_losses["auxiliary"] = float(loss)

    self.auxiliary_optimizer.step()

  def sample_data(self):
    data = ...
    try:
      data = next(self.data_iter)
    except StopIteration:
      # NOTE: with spawn, this pickles the dataset
      # for whatever reason.
      # Workaround: wrap datasets in infinite
      # length wrapper.
      self.data_iter = iter(self.data_loader)
      data = next(self.data_iter)

    return data

  def step(self):
    for _ in range(self.auxiliary_steps):
      data = self.sample_data()
      self.auxiliary_step(data)
    for _ in range(self.score_steps):
      data = self.sample_data()
      self.score_step(data)

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

  def each_generate(self, data):
    pass

  def report(self):
    sample = self.buffer.sample(self.batch_size)
    self.each_generate(sample)
    self.validate()

  def train(self):
    self.initialize()
    for _ in range(self.max_steps):
      self.step()
      self.log()
      self.step_id += 1

    self.finalize()

    return self.score
