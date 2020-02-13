import torch
import torch.nn as nn
import torch.nn.functional as func

from torch.utils.data import Dataset

from torchsupport.training.energy import EnergyTraining, SampleBuffer
from torchsupport.data.io import netwrite, to_device, make_differentiable
from torchsupport.data.collate import DataLoader, default_collate

class TransitionBuffer(SampleBuffer):
  def __init__(self, owner, buffer_size=10000):
    super().__init__(owner, buffer_size=buffer_size, buffer_probability=1.0)

  def __getitem__(self, idx):
    result = self.samples[idx % len(self.samples)]
    return result, idx % len(self.samples)

class EnergySamplerTraining(EnergyTraining):
  def __init__(self, score, sampler, *args,
               transition_buffer_size=10000, sampler_steps=10,
               sampler_optimizer=torch.optim.Adam, n_sampler=1,
               sampler_optimizer_kwargs=None,
               sampler_wrapper=lambda x: x,
               **kwargs):
    super().__init__(score, *args, **kwargs)
    self.transition_buffer = TransitionBuffer(self, transition_buffer_size)
    self.sampler = sampler.to(self.device)
    self.wrapper = sampler_wrapper(self.sampler)
    self.sampler_steps = sampler_steps
    self.n_sampler = n_sampler
    self.transition_buffer_loader = lambda x: DataLoader(
      x, batch_size=2 * self.batch_size, shuffle=True, drop_last=True
    )

    if sampler_optimizer_kwargs is None:
      sampler_optimizer_kwargs = {"lr": 5e-4}

    self.sampler_optimizer = sampler_optimizer(
      sampler.parameters(), **sampler_optimizer_kwargs
    )

  def sampler_step(self, data, idx):
    self.sampler_optimizer.zero_grad()
    data = to_device(data, self.device)
    args = self.run_sampler(data)
    loss_val = self.sampler_loss(*args)

    self.writer.add_scalar("sampler total loss", float(loss_val),  self.step_id * self.n_sampler + idx)

    loss_val.backward()
    self.sampler_optimizer.step()

  def sample_transition(self):
    with torch.no_grad():
      buffer_iter = iter(self.transition_buffer_loader(self.transition_buffer))
      (source, target), indices = to_device(next(buffer_iter), self.device)
      source, *source_args = source
      target, *target_args = target

      source_energy = self.score(source, *source_args)
      target_energy = self.score(target, *target_args)

      access = (target_energy < source_energy).view(-1)
      purge = (target_energy > source_energy).view(-1)

      purge_indices = indices[purge]
      try:
        for index in reversed(sorted(purge_indices)):
          del self.transition_buffer.samples[index]
      except:
        pass

    return (source, *source_args), (target, *target_args)

  def integrate(self, data, *args):
    transitions = []
    original = data.clone()
    for _ in range(self.sampler_steps):
      decrease = torch.ones((self.batch_size, 1), dtype=torch.float, device=self.device)
      modified = self.wrapper.sample(data, decrease, *args)
      data_score = self.score(data, *args)
      modified_score = self.score(modified, *args)
      access = (modified_score < data_score + 0.1 * abs(data_score.mean())).view(-1)
      #data[access] = modified[access]
      data = modified
    transitions.extend(
      [
        pair
        for pair in zip(
          self.decompose_batch(original.detach(), *args),
          self.decompose_batch(data.detach(), *args)
        )
      ]
    )
    self.transition_buffer.update(transitions)
    return data.detach()

  def sample(self):
    buffer_iter = iter(self.buffer_loader(self.buffer))
    data, *args = to_device(self.data_key(next(buffer_iter)), self.device)
    self.sampler.eval()
    with torch.no_grad():
      data = self.integrate(data, *args)
    self.sampler.train()
    detached = to_device(data.detach(), "cpu")
    update = self.decompose_batch(detached, *args)
    make_differentiable(update, toggle=False)
    self.buffer.update(update)

    return to_device((detached, *args), self.device)

  def sampler_condition(self, source, target):
    result = (target < source).float()
    return result

  def noise(self, data):
    result = data.clone()
    result[0.25 > torch.rand_like(result)] = 1
    result[0.25 > torch.rand_like(result)] = 0
    return result

  def run_sampler(self, data):
    data, *args = to_device(self.data_key(data), "cpu")
    first = self.noise(data)

    self.transition_buffer.update([
      pair
      for pair in zip(
        self.decompose_batch(first.detach(), *args),
        self.decompose_batch(data.detach(), *args)
      )
    ])

    source, target = self.sample_transition()
    source, *source_args = source
    target, *target_args = target
    with torch.no_grad():
      source_energy = self.score(source, *source_args)
      target_energy = self.score(target, *target_args)
    condition = self.sampler_condition(source_energy, target_energy)
    mask = (condition == 1).view(-1)
    prediction = self.sampler(source, condition, *source_args)
    return prediction, target, mask

  def sampler_loss(self, source, target, mask):
    return ((
      source[mask].clamp(0, 1) -
      target[mask].clamp(0, 1)
    ) ** 2).mean()

  def step(self, data):
    self.energy_step(data)
    for idx in range(self.n_sampler):
      buffer_iter = iter(self.buffer_loader(self.data))
      data = next(buffer_iter)
      self.sampler_step(data, idx)
    self.each_step()
