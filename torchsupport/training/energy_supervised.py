import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.data.io import netwrite, to_device, make_differentiable
from torchsupport.training.energy import EnergyTraining

class EnergySupervisedTraining(EnergyTraining):
  def logit_energy(self, logits):
    return -logits.logsumexp(dim=-1)

  def create_score(self):
    def _score(data, *args):
      logits = self.score(data, *args)
      return self.logit_energy(logits)
    return _score

  def sample(self):
    buffer_iter = iter(self.buffer_loader(self.buffer))
    data, *args = to_device(self.data_key(next(buffer_iter)), self.device)
    self.score.eval()
    data = self.integrator.integrate(self.create_score(), data, *args).detach()
    self.score.train()
    detached = to_device(data.detach(), "cpu")
    update = (to_device((detached[idx], *[arg[idx] for arg in args]), "cpu") for idx in range(data.size(0)))
    make_differentiable(update, toggle=False)
    self.buffer.update(update)

    return to_device((detached, *args), self.device)

  def classifier_loss(self, logits, labels):
    return func.cross_entropy(logits, labels)

  def run_energy(self, data):
    data, labels = data
    real_logits, fake_logits = super().run_energy(data)
    real_result = self.logit_energy(real_logits)
    fake_result = self.logit_energy(fake_logits)

    # set integrator target, if appropriate.
    if self.integrator.target is None:
      self.integrator.target = real_result.detach().mean(dim=0)
    self.integrator.target = 0.6 * self.integrator.target + 0.4 * real_result.detach().mean(dim=0)
    return real_result, fake_result, real_logits, labels

  def energy_loss(self, real_result, fake_result, real_logits, labels):
    energy = super().energy_loss(real_result, fake_result)
    classifier = self.classifier_loss(real_logits, labels)
    self.current_losses["classifier loss"] = float(classifier)
    return energy + classifier

class EnergyConditionalTraining(EnergySupervisedTraining):
  def logit_energy(self, logits, labels):
    return logits[torch.arange(logits.size(0)), labels]

  def create_score(self):
    def _score(data, *args):
      labels = args[0]
      logits = self.score(data, *args)
      return self.logit_energy(logits, labels)
    return _score

  def classifier_loss(self, logits, labels):
    return func.cross_entropy(-logits, labels)

  def run_energy(self, data):
    data, labels = data
    fake, fake_labels = self.sample()

    if self.step_id % self.report_interval == 0:
      detached, *args = self.data_key(to_device((fake, fake_labels), "cpu"))
      self.each_generate(detached.detach(), *args)

    make_differentiable(fake)
    make_differentiable(data)
    input_data, *data_args = self.data_key((data, labels))
    input_fake, *fake_args = self.data_key((fake, fake_labels))
    real_logits = self.score(input_data, *data_args)
    fake_logits = self.score(input_fake, *fake_args)
    real_result = self.logit_energy(real_logits, labels)
    fake_result = self.logit_energy(fake_logits, fake_labels)
    return real_result, fake_result, real_logits, labels
