from torch.distributions import Normal, RelaxedOneHotCategorical

from tensorboardX import SummaryWriter

from torchsupport.training.state import (
  NetState, NetNameListState, TrainingState
)
from torchsupport.training.training import Training
import torchsupport.modules.losses.vae as vl
from torchsupport.structured import DataParallel as SDP
from torchsupport.data.io import netwrite, to_device, detach, make_differentiable
from torchsupport.data.collate import DataLoader
from torchsupport.modules.losses.generative import normalized_diversity_loss

class StepDescriptor:
  def __init__(self, name="step", n_steps="n_steps",
               n_steps_default=1, every="every",
               every_default=1):
    self.name = name
    self.n_steps = n_steps
    self.n_steps_value = n_steps_default
    self.every = every
    self.every_value = every_default

  def step(self, target, data):
    to_run = getattr(target, self.name)
    return to_run(data)

def step_descriptor(n_steps="n_steps", n_steps_default=1,
                    every="every", every_default=1):
  def helper(method):
    name = method.__name__
    result = StepDescriptor(
      name, n_steps=n_steps, n_steps_default=n_steps_default,
      every=every, every_default=every_default
    )
    method._step_descriptor = result
    return method
  return helper

class StepRegistry(type):
  def __init__(cls, name, bases, attrs):
    sds = dict()
    for name, method in attrs.items():
      if hasattr(method, "_step_descriptor"):
        sds[name] = method._step_descriptor

    @property
    def step_descriptors(self):
      tags = sds.copy()
      try:
        tags.update(super(cls, self).step_descriptors)
      except AttributeError:
        pass
      return tags

    cls.step_descriptors = step_descriptors

class MultistepTraining(Training, metaclass=StepRegistry):
  """Abstract base class for multi-step training."""
  checkpoint_parameters = Training.checkpoint_parameters + [
    TrainingState()
  ]
  step_descriptors = {}
  step_order = []
  def __init__(self, networks, network_mapping, data, **kwargs):
    super().__init__(**kwargs)

    self.data = {}
    self.loaders = {}
    self.optimizers = {}
    for name, descriptor in self.step_descriptors.items():
      descriptor.n_steps_value = kwargs.get(descriptor.n_steps) or descriptor.n_steps_value
      descriptor.every_value = kwargs.get(descriptor.every) or descriptor.every_value
      self.optimizers[name] = []
      self.data[name] = None
      self.loaders[name] = None

    self.nets = {}
    for name, (network, opt, opt_kwargs) in networks.items():
      network = network.to(self.device)
      optimizer = opt(network.parameters(), **opt_kwargs)
      optimizer_name = f"{name}_optimizer"
      setattr(self, name, network)
      setattr(self, optimizer_name, optimizer)
      self.checkpoint_parameters += [
        NetState(name), NetState(optimizer_name)
      ]
      self.checkpoint_names.update({
        name: network
      })
      self.nets[name] = (network, optimizer)
    for step, names in network_mapping.items():
      self.optimizers[step] = [
        self.nets[name][1]
        for name in names
      ]

    for step, data_set in data.items():
      self.data[step] = data_set
      self.loaders[step] = None

    self.current_losses = {}

  def get_data(self, step):
    if self.data[step] is None:
      return None
    data = self.loaders[step]
    if data is None:
      data = iter(DataLoader(
        self.data[step], batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=True, drop_last=True
      ))
      self.loaders[step] = data
    try:
      data_point = to_device(next(data), self.device)
    except StopIteration:
      data = iter(DataLoader(
        self.data[step], batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=True, drop_last=True
      ))
      self.loaders[step] = data
      data_point = to_device(next(data), self.device)
    return data_point

  def step(self):
    total_loss = 0.0
    for step_name in self.step_order:
      descriptor = self.step_descriptors[step_name]
      if self.step_id % descriptor.every_value == 0:
        for _ in range(descriptor.n_steps_value):
          data = self.get_data(step_name)
          for optimizer in self.optimizers[step_name]:
            optimizer.zero_grad()
          loss = descriptor.step(self, data)
          if isinstance(loss, (list, tuple)):
            loss, *_ = loss
          if loss is not None:
            loss.backward()
          for optimizer in self.optimizers[step_name]:
            optimizer.step()
        total_loss += float(loss)
    self.log_statistics(total_loss)
    self.each_step()

  def train(self):
    for step_id in range(self.max_steps):
      self.step_id += step_id
      self.step()
      self.log()

    return self.nets
