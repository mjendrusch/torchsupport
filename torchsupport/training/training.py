import os
import time
import random
from copy import copy

import numpy as np
import torch

from tensorboardX import SummaryWriter

from torchsupport.data.io import netread, netwrite, to_device
from torchsupport.data.episodic import SupportData
from torchsupport.data.collate import DataLoader

from torchsupport.training.state import (
  TrainingState, NetState, State, SaveStateError
)

class Training(object):
  """Abstract training process class."""
  checkpoint_parameters = []
  torch_rng_state = torch.random.get_rng_state()
  np_rng_state = np.random.get_state()
  random_rng_state = random.getstate()

  save_interval = 600
  last_tick = 0

  def __init__(self,
               max_epochs=50,
               max_steps=int(1e7),
               batch_size=128,
               num_workers=8,
               device="cpu",
               path_prefix=".",
               network_name="network",
               verbose=False,
               report_interval=10,
               checkpoint_interval=1000):
    self.max_epochs = max_epochs
    self.max_steps = max_steps
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.device = device
    self.path_prefix = path_prefix
    self.network_name = network_name
    self.full_path = f"{self.path_prefix}/{self.network_name}"
    self.verbose = verbose
    self.report_interval = report_interval
    self.checkpoint_interval = checkpoint_interval
    self.checkpoint_names = {}
    self.step_id = 0
    self.epoch_id = 0
    self.writer = SummaryWriter(self.full_path)
    self.current_losses = {}

  def collect_netlist(self, networks):
    netlist = []
    names = []
    for name, network in networks.items():
      names.append(name)
      network_object = network.to(self.device)
      setattr(self, name, network_object)
      netlist.extend(list(network_object.parameters()))
    return names, netlist

  def get_netlist(self, netlist):
    return {
      name: getattr(self, name)
      for name in netlist
    }

  def log_statistics(self, loss_val, prefix="", suffix=" loss", name="total loss"):
    if self.verbose:
      for loss_name in self.current_losses:
        loss_float = self.current_losses[loss_name]
        self.writer.add_scalar(f"{prefix}{loss_name}{suffix}", loss_float, self.step_id)
    self.writer.add_scalar(name, float(loss_val), self.step_id)

  def each_step(self):
    self.save_tick()

  def each_validate(self):
    pass

  def each_epoch(self):
    pass

  def each_checkpoint(self):
    pass

  def train(self):
    pass

  def validate(self):
    pass

  def save_path(self):
    return f"{self.full_path}-save.torch"

  def run_checkpoint(self):
    for name, the_net in self.checkpoint_names.items():
      if isinstance(the_net, torch.nn.DataParallel):
        the_net = the_net.module
      netwrite(
        the_net,
        f"{self.full_path}-{name}-epoch-{self.epoch_id}-step-{self.step_id}.torch"
      )
    self.each_checkpoint()

  def emergency_read_checkpoint(self):
    import glob
    for name, the_net in self.checkpoint_names.items():
      if isinstance(the_net, torch.nn.DataParallel):
        the_net = the_net.module
      files = glob.glob(f"{self.full_path}-{name}-epoch-*.torch")
      files = sorted(files, key=lambda x: int(x.split("-")[-1].split(".")[0]))
      target = files[-1]
      netread(
        the_net,
        target
      )

  def run_report(self):
    pass

  def checkpoint(self):
    if self.step_id % self.checkpoint_interval == 0:
      self.run_checkpoint()

  def report(self):
    if self.step_id % self.report_interval == 0:
      self.run_report()

  def log(self):
    self.checkpoint()
    self.report()

  def write(self, path):
    data = {}
    data["_torch_rng_state"] = torch.random.get_rng_state()
    data["_np_rng_state"] = np.random.get_state()
    data["_random_rng_state"] = random.getstate()
    for param in self.checkpoint_parameters:
      param.write_action(self, data)
    torch.save(data, path + ".tmp")
    if os.path.isfile(path):
      os.rename(path, path + ".old")
    os.rename(path + ".tmp", path)

  def read(self, path):
    data = torch.load(path)
    torch.random.set_rng_state(data["_torch_rng_state"])
    np.random.set_state(data["_np_rng_state"])
    random.setstate(data["_random_rng_state"])
    for param in self.checkpoint_parameters:
      param.read_action(self, data)

  def save(self, path=None):
    path = path or self.save_path()
    self.write(path)

  def save_tick(self, step=None):
    step = step or self.save_interval
    this_tick = time.monotonic()
    if this_tick - self.last_tick > step:
      try:
        self.save()
        self.last_tick = this_tick
      except SaveStateError:
        torch_rng_state = torch.random.get_rng_state()
        np_rng_state = np.random.get_state()
        random_rng_state = random.getstate()
        self.load()
        torch.random.set_rng_state(torch_rng_state)
        np.random.set_state(np_rng_state)
        random.setstate(random_rng_state)

  def load(self, path=None):
    try:
      path = path or self.save_path()
      if os.path.isfile(path):
        self.read(path)
    except Exception:
      print("Something went wrong! Trying to read latest network checkpoints...")
      self.emergency_read_checkpoint()
    return self

class SupervisedTraining(Training):
  """Standard supervised training process.

  Args:
    net (Module): a trainable network module.
    train_data (DataLoader): a :class:`DataLoader` returning the training
                              data set.
    validate_data (DataLoader): a :class:`DataLoader` return ing the
                                 validation data set.
    optimizer (Optimizer): an optimizer for the network. Defaults to ADAM.
    schedule (Schedule): a learning rate schedule. Defaults to decay when
                          stagnated.
    max_epochs (int): the maximum number of epochs to train.
    device (str): the device to run on.
    checkpoint_path (str): the path to save network checkpoints.
  """
  checkpoint_parameters = Training.checkpoint_parameters + [
    TrainingState(),
    NetState("net"),
    NetState("optimizer")
  ]
  def __init__(self, net, train_data, validate_data, losses,
               optimizer=torch.optim.Adam,
               schedule=None,
               accumulate=None,
               valid_callback=None,
               **kwargs):
    super(SupervisedTraining, self).__init__(**kwargs)
    self.valid_callback = valid_callback or (lambda x, y, z: None)
    self.accumulate = accumulate
    self.optimizer = optimizer(net.parameters())
    if schedule is None:
      self.schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)
    else:
      self.schedule = schedule
    self.losses = losses
    self.train_data = DataLoader(
      train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True
    )
    self.validate_data = DataLoader(
      validate_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True
    )
    self.valid_iter = iter(self.validate_data)
    self.net = net.to(self.device)

    self.checkpoint_names = dict(checkpoint=self.net)
    self.validation_losses = [0 for _ in range(len(self.losses))]
    self.training_losses = [0 for _ in range(len(self.losses))]
    self.best = None

  def run_networks(self, data):
    inputs, *labels = data
    if not isinstance(inputs, (list, tuple)):
      inputs = [inputs]
    predictions = self.net(*inputs)
    if not isinstance(predictions, (list, tuple)):
      predictions = [predictions]
    return [combined for combined in zip(predictions, labels)]

  def loss(self, inputs):
    loss_val = torch.tensor(0.0).to(self.device)
    for idx, the_input in enumerate(inputs):
      this_loss_val = self.losses[idx](*the_input)
      self.training_losses[idx] = float(this_loss_val)
      loss_val += this_loss_val
    return loss_val

  def valid_loss(self, inputs):
    training_cache = list(self.training_losses)
    loss_val = self.loss(inputs)
    self.validation_losses = self.training_losses
    self.training_losses = training_cache
    return loss_val

  def chunk(self, data, split):
    if torch.is_tensor(data):
      return data.split(len(data) // split)
    elif isinstance(data, (list, tuple)):
      result = [
        [] for idx in range(split)
      ]
      for item in data:
        for target, part in zip(result, self.chunk(item, split)):
          target.append(part)
      return result
    elif isinstance(data, dict):
      result = [
        {}
        for idx in range(split)
      ]
      for name in data:
        for dd, part in zip(result, self.chunk(data[name], split)):
          dd[name] = part
      return result
    else:
      return data

  def step(self, data):
    self.optimizer.zero_grad()
    if self.accumulate is not None:
      points = self.chunk(data, self.accumulate)
      for point in points:
        outputs = self.run_networks(point)
        loss_val = self.loss(outputs) / self.accumulate
        loss_val.backward()
    else:
      outputs = self.run_networks(data)
      loss_val = self.loss(outputs)
      loss_val.backward()
    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
    self.optimizer.step()
    self.each_step()

  def validate(self, data):
    with torch.no_grad():
      self.net.eval()
      if self.accumulate is not None:
        point = self.chunk(data, self.accumulate)[0]
        outputs = self.run_networks(point)
      else:
        outputs = self.run_networks(data)
      self.valid_loss(outputs)
      self.each_validate()
      self.valid_callback(
        self, to_device(data, "cpu"), to_device(outputs, "cpu")
      )
      self.net.train()

  def schedule_step(self):
    self.schedule.step(sum(self.validation_losses))

  def each_step(self):
    Training.each_step(self)
    for idx, loss in enumerate(self.training_losses):
      self.writer.add_scalar(f"training loss {idx}", loss, self.step_id)
    self.writer.add_scalar(f"training loss total", sum(self.training_losses), self.step_id)

  def each_validate(self):
    for idx, loss in enumerate(self.validation_losses):
      self.writer.add_scalar(f"validation loss {idx}", loss, self.step_id)
    self.writer.add_scalar(f"validation loss total", sum(self.validation_losses), self.step_id)

  def run_report(self):
    vdata = None
    try:
      vdata = next(self.valid_iter)
    except StopIteration:
      self.valid_iter = iter(self.validate_data)
      vdata = next(self.valid_iter)
    vdata = to_device(vdata, self.device)
    self.validate(vdata)

  def train(self):
    for epoch_id in range(self.max_epochs):
      self.epoch_id = epoch_id
      for data in self.train_data:
        data = to_device(data, self.device)
        self.step(data)
        self.log()
        self.step_id += 1
      self.schedule_step()
      self.each_epoch()
    return self.net

class MaskedSupervisedTraining(SupervisedTraining):
  def run_networks(self, data):
    inputs, labels_masks = data
    labels = [label for (label, mask) in labels_masks]
    masks = [mask for (label, mask) in labels_masks]
    predictions = self.net(inputs)
    return list(zip(predictions, labels, masks))

class FewShotTraining(SupervisedTraining):
  def __init__(self, net, train_data, validate_data, losses,
               **kwargs):
    super(FewShotTraining, self).__init__(
      net, train_data, validate_data, losses,
      **kwargs
    )

    support_data = copy(train_data)
    train_data.data_mode = type(train_data.data_mode)(1)
    support_data = SupportData(train_data, shots=5)
    validate_support_data = SupportData(validate_data, shots=5)
    self.support_loader = iter(DataLoader(support_data))
    self.valid_support_loader = iter(DataLoader(validate_support_data))

  def run_networks(self, data, support, support_label):
    predictions = self.net(data, support)
    return list(zip(predictions, support_label))

  def step(self, inputs):
    data, label = inputs
    self.optimizer.zero_grad()

    permutation = [0, 1, 2]
    random.shuffle(permutation)

    support, support_label = next(self.support_loader)

    lv = label[0].reshape(-1)
    for idx, val in enumerate(lv):
      lv[idx] = permutation[int(val[0])]
    lv = support_label.reshape(-1)
    for idx, val in enumerate(lv):
      lv[idx] = permutation[int(val[0])]

    support = support[0].to(self.device)
    support_label = support_label[0].to(self.device)
    outputs = self.run_networks(data, support, support_label)

    loss_val = self.loss(outputs)
    loss_val.backward()
    self.optimizer.step()
    self.each_step()  

  def validate(self, data):
    with torch.no_grad():
      self.net.eval()
      inputs, *label = data
      inputs, label = inputs.to(self.device), list(map(lambda x: x.to(self.device), label))
      support, support_label = next(self.valid_support_loader)
      support = support[0].to(self.device)
      support_label = support_label[0].to(self.device)
      outputs = self.run_networks(inputs, support, support_label)
      self.valid_loss(outputs)
      self.each_validate()
      self.valid_callback(self, to_device(inputs, "cpu"), to_device(outputs, "cpu"))
      self.net.train()
