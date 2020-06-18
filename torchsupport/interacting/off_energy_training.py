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

from torchsupport.data.collate import DataLoader

from torchsupport.interacting.buffer import SchemaBuffer
from torchsupport.interacting.collector_task import EnergyCollector
from torchsupport.interacting.distributor_task import DefaultDistributor
from torchsupport.interacting.data_collector import ExperienceCollector
from torchsupport.interacting.stats import EnergyStatistics

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
               max_steps=1_000_000,
               buffer_size=100_000,
               score_steps=1,
               auxiliary_steps=1,
               n_workers=8,
               batch_size=64,
               checkpoint_interval=10,
               optimizer=torch.optim.Adam,
               optimizer_kwargs=None,
               aux_optimizer=torch.optim.Adam,
               aux_optimizer_kwargs=None,
               device="cpu",
               network_name="network",
               path_prefix=".",
               report_interval=1000,
               verbose=True):
    self.epoch_id = 0
    self.max_steps = max_steps
    self.score_steps = score_steps
    self.auxiliary_steps = auxiliary_steps
    self.checkpoint_interval = checkpoint_interval
    self.report_interval = report_interval
    self.step_id = 0
    self.verbose = verbose
    self.checkpoint_path = f"{path_prefix}/{network_name}-checkpoint"

    self.current_losses = {}
    self.writer = SummaryWriter(f"{path_prefix}/{network_name}")

    self.statistics = EnergyStatistics()
    self.device = device
    self.batch_size = batch_size
    self.integrator = integrator
    self.decay = decay
    self.score = score.to(device)
    self.energy = energy
    self.collector = EnergyCollector(energy, integrator, self.batch_size)
    self.distributor = DefaultDistributor()
    self.data_collector = ExperienceCollector(
      self.distributor, self.collector, n_workers=n_workers
    )
    self.buffer = SchemaBuffer(self.data_collector.schema(), buffer_size)

    self.data = data
    self.data_loader = DataLoader(
      self.data, batch_size=self.batch_size, num_workers=8,
      shuffle=True, drop_last=True
    )
    self.data_iter = iter(self.data_loader)

    optimizer_kwargs = optimizer_kwargs or {}
    self.optimizer = optimizer(
      self.score.parameters(), **optimizer_kwargs
    )

    auxiliary_netlist = []
    self.auxiliary_names = []
    for network in auxiliary_networks:
      self.auxiliary_names.append(network)
      network_object = auxiliary_networks[network].to(device)
      setattr(self, network, network_object)
      auxiliary_netlist.extend(list(network_object.parameters()))

    aux_optimizer_kwargs = aux_optimizer_kwargs or {}
    self.auxiliary_optimizer = aux_optimizer(
      auxiliary_netlist, **aux_optimizer_kwargs
    )

  def save_path(self):
    return self.checkpoint_path + "-save"

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

  def checkpoint(self):
    the_net = self.score
    if isinstance(self.score, torch.nn.DataParallel):
      the_net = the_net.module
    netwrite(
      the_net,
      f"{self.checkpoint_path}-step-{self.step_id}.torch"
    )
    self.each_checkpoint()

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

  def train(self):
    self.initialize()
    for _ in range(self.max_steps):
      self.step()
      if self.step_id % self.report_interval == 0:
        sample = self.buffer.sample()
        self.each_generate(sample)
        self.validate()
      if self.step_id % self.checkpoint_interval == 0:
        self.checkpoint()
      self.step_id += 1

    self.finalize()

    return self.score
