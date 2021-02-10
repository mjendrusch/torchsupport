from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Categorical

from torchsupport.training.state import (
  NetNameListState, TrainingState
)
from torchsupport.training.multistep_training import MultistepTraining, step_descriptor
from torchsupport.data.io import to_device, make_differentiable
from torchsupport.data.collate import DataLoader, default_collate
from torchsupport.data.match import match

class TiltedSupervisedTraining(MultistepTraining):
  step_order = ["network_step", "policy_step"]
  def __init__(self, network, policy, data,
               entropy_weight=1e-4, optimizer=None,
               network_optimizer_kwargs=None,
               policy_optimizer_kwargs=None,
               network_options=None,
               mapping_options=None,
               data_options=None,
               **kwargs):
    optimizer = optimizer or torch.optim.Adam
    network_optimizer_kwargs = network_optimizer_kwargs or {}
    policy_optimizer_kwargs = policy_optimizer_kwargs or {}
    network_options = network_options or {}
    mapping_options = mapping_options or {}
    data_options = data_options or {}

    networks = dict(
      network=(network, optimizer, network_optimizer_kwargs),
      policy=(policy, optimizer, policy_optimizer_kwargs)
    )
    networks.update(network_options)

    mapping = dict(
      policy_step=["policy"],
      network_step=["network"]
    )
    mapping.update(mapping_options)

    data = dict(
      policy_step=data,
      network_step=data
    )
    data.update(data_options)

    self.network = ...
    self.policy = ...
    super().__init__(
      networks, mapping, data, **kwargs
    )
    self.entropy_weight = entropy_weight

  def sample(self, data, scale=1.0):
    # sample from the policy
    sample_size = data.size(1)
    data = data.view(-1, *data.shape[2:])
    policy = self.policy(data)
    policy = policy.view(-1, sample_size).log_softmax(dim=1)
    samples = Categorical(logits=policy * scale).sample().view(-1)
    ind = torch.arange(0, policy.size(0), dtype=torch.long, device=policy.device)
    data = data.view(-1, sample_size, *data.shape[1:])
    data = data[ind, samples]
    return data, policy[ind, samples]

  def run_policy(self, data):
    data, labels = data
    data, policy = self.sample(data)

    with torch.no_grad():
      predictions = self.network(data)

    self.writer.add_images("policy images", data.detach().cpu(), self.step_id)

    return policy, predictions, labels

  def policy_loss(self, policy, predictions, labels):
    total_loss = 0.0
    if isinstance(labels, (list, tuple)):
      policy = policy.chunk(len(labels), dim=1)
      for idx, (pi, pred, label) in enumerate(zip(
          policy, predictions, labels
      )):
        reward = -self.loss(pred, label)
        value = (reward).mean().detach()
        advantage = reward - value
        loss = pi.log_softmax(dim=0) * advantage.exp().clamp(0, 20)
        loss = -loss.mean()
        self.current_losses[f"task policy {idx}"] = float(loss)
        total_loss += loss
    else:
      # policy = policy[:, 0]
      reward = -self.loss(predictions, labels)
      value = (policy.exp() * reward).mean().detach()
      self.current_losses[f"value"] = float(value)
      advantage = 10 * (reward - value)
      total_loss = policy * advantage.exp().clamp(0, 20)
      total_loss = -total_loss.mean()
      self.current_losses["policy"] = float(total_loss)

    return total_loss

  @step_descriptor(n_steps="n_policy", every="every_policy")
  def policy_step(self, data):
    args = self.run_policy(data)
    return self.policy_loss(*args)

  def loss(self, prediction, target):
    return func.cross_entropy(
      prediction, target, reduction="none"
    )

  def run_network(self, data):
    data, labels = data
    with torch.no_grad():
      data, policy = self.sample(data, scale=5)
    self.writer.add_images("network images", data.detach().cpu(), self.step_id)

    predictions = self.network(data)
    return policy, predictions, labels

  def network_loss(self, policy, predictions, labels):
    total_loss = 0.0
    if isinstance(labels, (list, tuple)):
      policy = policy.chunk(len(labels), dim=1)
      for idx, (pi, pred, label) in enumerate(zip(
          policy, predictions, labels
      )):
        loss = self.loss(pred, label)
        loss = pi.softmax(dim=0) * loss
        loss = loss.sum(dim=0)
        self.current_losses[f"task {idx}"] = float(loss)
        total_loss += loss
    else:
      total_loss = self.loss(predictions, labels)
      # total_loss = policy.exp() * total_loss
      total_loss = total_loss.mean()
      self.current_losses["network"] = float(total_loss)

    return total_loss

  @step_descriptor(n_steps="n_network", every="every_network")
  def network_step(self, data):
    args = self.run_network(data)
    return self.network_loss(*args)
