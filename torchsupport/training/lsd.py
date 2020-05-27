import random
from copy import deepcopy
from itertools import islice

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.autograd as ag

from torchsupport.data.io import netwrite, to_device, make_differentiable
from torchsupport.data.collate import DataLoader, default_collate
from torchsupport.training.energy import AbstractEnergyTraining
from torchsupport.training.samplers import AnnealedLangevin

class LSDTraining(AbstractEnergyTraining):
  def __init__(self, score, critic, *args,
               decay=0.1, integrator=None,
               optimizer=torch.optim.Adam,
               n_critic=5, critic_optimizer_kwargs=None,
               **kwargs):
    self.score = ...
    self.critic = ...
    super().__init__(
      {"score": score}, *args, optimizer=optimizer, **kwargs
    )

    critics = {"critic": critic}
    netlist = []
    self.critic_names = []
    for network in critics:
      self.critic_names.append(network)
      network_object = critics[network].to(self.device)
      setattr(self, network, network_object)
      netlist.extend(list(network_object.parameters()))

    if critic_optimizer_kwargs is None:
      critic_optimizer_kwargs = {"lr": 5e-4}

    self.critic_data = DataLoader(
      self.data, batch_size=self.batch_size, num_workers=8,
      shuffle=True, drop_last=True
    )
    self.critic_optimizer = optimizer(
      netlist,
      **critic_optimizer_kwargs
    )

    self.n_critic = n_critic
    self.decay = decay
    self.integrator = integrator

  def step(self, data):
    for critic_data in islice(self.critic_data, self.n_critic):
      self.critic_step(critic_data)
    super().step(data)

  def critic_step(self, data):
    """Performs a single step of critic training.

    Args:
      data: data points used for training.
    """
    self.critic_optimizer.zero_grad()
    data = to_device(data, self.device)
    args = self.run_energy(data)
    loss_val = self.critic_loss(*args)

    self.current_losses["critic total"] = float(loss_val)

    loss_val.backward()
    self.critic_optimizer.step()

  def each_step(self):
    super().each_step()
    if self.step_id % self.report_interval == 0 and self.step_id != 0:
      data, *args = self.sample()
      self.each_generate(data, *args)

  def noise_vectors(self, score):
    return torch.randn_like(score)

  def data_key(self, data):
    result, *args = data
    return (result, *args)

  def critic_loss(self, data, score, critic):
    result = self.energy_loss(data, score, critic)
    l2_loss = (critic.view(critic.size(0), -1) ** 2).sum(dim=-1).mean()
    return -result + self.decay * l2_loss

  def energy_loss(self, data, score, critic):
    vectors = self.noise_vectors(critic)
    grad_score = ag.grad(
      score, data,
      grad_outputs=torch.ones_like(score),
      create_graph=True
    )[0]
    jacobian = ag.grad(
      critic, data,
      grad_outputs=vectors,
      create_graph=True
    )[0]
    jacobian_term = (vectors * jacobian).view(score.size(0), -1).sum(dim=-1)
    critic_term = (grad_score * critic).view(score.size(0), -1).sum(dim=-1)

    penalty_term = (score ** 2).mean()

    self.current_losses["jacobian"] = float(jacobian_term.mean())
    self.current_losses["critic"] = float(critic_term.mean())
    self.current_losses["penalty"] = float(penalty_term.mean())

    return (jacobian_term + critic_term).mean()

  def run_energy(self, data):
    data, *args = self.data_key(data)
    make_differentiable(data)
    critic = self.critic(data, *args)
    score = self.score(data, *args)
    return data, score, critic

  def prepare_sample(self):
    results = []
    for idx in range(self.batch_size):
      results.append(self.prepare())
    return default_collate(results)

  def sample(self):
    scr = self.score
    self.score.eval()
    integrator = self.integrator
    prep = to_device(self.prepare_sample(), self.device)
    data, *args = self.data_key(prep)
    result = integrator.integrate(
      scr,
      data, *args
    ).detach()
    self.score.train()
    return to_device((result, data, *args), self.device)
