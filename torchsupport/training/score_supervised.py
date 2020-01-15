import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.data.io import netwrite, to_device, make_differentiable
from torchsupport.training.energy import DenoisingScoreTraining
from torchsupport.training.samplers import AnnealedLangevin

class ScoreSupervisedTraining(DenoisingScoreTraining):
  def logit_energy(self, logits):
    return -logits.logsumexp(dim=-1)

  def create_score(self):
    def _score(data, sigma, *args):
      score, logits = self.score(data, sigma, *args)
      return score
    return _score

  def classifier_loss(self, logits, labels):
    return func.cross_entropy(logits, labels)

  def sample(self):
    self.score.eval()
    with torch.no_grad():
      integrator = AnnealedLangevin([
        self.sigma * self.factor ** idx for idx in range(self.n_sigma)
      ])
      prep = to_device(self.prepare_sample(), self.device)
      data, *args = self.data_key(prep)
      result = integrator.integrate(
        self.create_score(),
        data, *args
      ).detach()
    self.score.train()
    return to_device((result, data, *args), self.device)

  def run_energy(self, data):
    data, labels = data
    data, *args = self.data_key(data)
    noisy, sigma = self.noise(data)
    score, logits = self.score(noisy, sigma, *args)

    return score, data, noisy, sigma, logits, labels

  def energy_loss(self, score, data, noisy, sigma, logits, labels):
    energy = super().energy_loss(score, data, noisy, sigma)
    classifier = self.classifier_loss(logits, labels)
    self.current_losses["classifier"] = float(classifier)
    return energy + classifier
