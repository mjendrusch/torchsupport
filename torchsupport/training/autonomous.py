import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.data.io import netwrite

class Training(object):
  """Abstract training process class.
  """
  def __init__(self):

  def each_step(self, step_id):
    pass

  def each_epoch(self, epoch_id):
    pass

  def each_checkpoint(self):
    pass

  def train(self):
    pass

  def validate(self):
    pass

class BasicTraining(Training):
  """Standard supervised training process.

  Arguments:
    net (Module) : a trainable network module.
    train_data (DataLoader) : a :class:`DataLoader` returning the training
                              data set.
    validate_data (DataLoader) : a :class:`DataLoader` return ing the
                                 validation data set.
    optimizer (Optimizer) : an optimizer for the network. Defaults to ADAM.
    schedule (Schedule) : a learning rate schedule. Defaults to decay when
                          stagnated.
    max_epochs (int) : the maximum number of epochs to train.
    device (str) : the device to run on.
    checkpoint_path (str) : the path to save network checkpoints.
  """
  def __init__(self, net, train_data, validate_data,
               optimizer=torch.optim.Adam(),
               schedule=None,
               max_epochs=10000,
               device=None,
               checkpoint_path="checkpoint.torch"):
    self.device = device
    self.optimizer = optimizer
    if schedule == None:
      self.schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
    else:
      self.schedule = schedule
    self.train_data = train_data
    self.validate_data = validate_data
    self.net = net
    self.max_epochs = max_epochs
    self.checkpoint_path = checkpoint_path
    self.step_id = 0
    self.epoch_id = 0
    self.validation_loss = 0
    self.training_loss = 0
    self.best = None

  def checkpoint(self):
    netwrite(self.net, self.checkpoint_path)

  def step(self, data, label, step_id, epoch_id):
    predictions = self.net(data)
    loss_val = self.loss(predictions, label)
    loss_val.backwards()
    self.optimizer.step()
    self.training_loss = loss_val.item()

  def validate(self):
    vit = iter(self.validate_data)
    vinputs, vlabels = next(vit).values()
    vinputs, vlabels = vinputs.to(device), vlabels.to(device)
    voutputs = self.net(vinputs)
    vloss_val = self.loss(voutputs, vlabels)
    self.validation_loss = vloss_val.item()

  def schedule_step(self):
    self.schedule.step(self.validation_loss)

  def each_step(self):
    if self.best == None or self.validation_loss < self.best:
      self.best = self.validation_loss
      self.checkpoint()

  def train(self):
    for epoch_id in range(max_epochs):
      self.epoch_id = epoch_id
      validation_loss = None
      for data in self.train_data:
        input, label = data.values()
        self.step(input, label)
        self.validate()
        self.each_step()
        self.step_id += 1
      self.schedule_step()
      self.each_epoch()

class RelabelTraining(BasicTraining):
  """Relabelling training process.

  Arguments:
    net (Module) : a trainable network module.
    train_data (DataLoader) : a :class:`DataLoader` returning the training
                              data set.
    validate_data (DataLoader) : a :class:`DataLoader` return ing the
                                 validation data set.
    optimizer (Optimizer) : an optimizer for the network. Defaults to ADAM.
    schedule (Schedule) : a learning rate schedule. Defaults to decay when
                          stagnated.
    max_epochs (int) : the maximum number of epochs to train.
    device (str) : the device to run on.
    checkpoint_path (str) : the path to save network checkpoints.
    relabel_steps (int) : number of steps after which the labels are
                          recomputed.
  """
  def __init__(self, net, train_data, validate_data,
               optimizer=torch.optim.Adam(),
               schedule=None,
               max_epochs=10000,
               device=None,
               checkpoint_path="checkpoint.torch",
               relabel_steps=2):
    super(RelabelTraining, self).__init__(
      net, train_data, validate_data,
      optimizer=optimizer,
      schedule=schedule,
      max_epochs=max_epochs,
      device=device,
      checkpoint_path=checkpoint_path
    )
    self.relabel_net = None
    self.relabel_steps = relabel_steps

  def relabel(self):
    self.relabel_net = copy.deepcopy(self.net)
    self.relabel_net = self.relabel_net.to(self.device)

  def each_relabel(self):
    pass

  def validate(self):
    vit = iter(self.validate_data)
    vinputs, vlabels = next(vit).values()
    vinputs, vlabels = vinputs.to(device), vlabels.to(device)
    if self.relabel_net != None:
      vlabels = self.relabel_net(vinputs)
    voutputs = self.net(vinputs)
    vloss_val = self.loss(voutputs, vlabels)
    self.validation_loss = vloss_val.item()

  def train(self):
    for epoch_id in range(max_epochs):
      self.epoch_id = epoch_id
      validation_loss = None
      for data in self.train_data:
        input, label = data.values()
        if self.relabel_net != None:
          label = self.relabel_net(input)
        self.step(input, label)
        self.validate()
        self.each_step()
        self.step_id += 1
      if (self.epoch_id + 1) % self.relabel_steps:
        self.relabel()
        self.each_relabel()
      self.schedule_step()
      self.each_epoch()

class LabelSpillTraining(BasicTraining):
  """Relabelling training process using a class-wise spill matrix.

  Arguments:
    net (Module) : a trainable network module.
    train_data (DataLoader) : a :class:`DataLoader` returning the training
                              data set.
    validate_data (DataLoader) : a :class:`DataLoader` return ing the
                                 validation data set.
    optimizer (Optimizer) : an optimizer for the network. Defaults to ADAM.
    schedule (Schedule) : a learning rate schedule. Defaults to decay when
                          stagnated.
    max_epochs (int) : the maximum number of epochs to train.
    device (str) : the device to run on.
    checkpoint_path (str) : the path to save network checkpoints.
    relabel_steps (int) : number of steps after which the labels are
                          recomputed.
  """
  def __init__(self, net, train_data, validate_data,
               optimizer=torch.optim.Adam(),
               relabel_data=None,
               schedule=None,
               max_epochs=10000,
               device=None,
               checkpoint_path="checkpoint.torch",
               relabel_steps=2):
    super(LabelSpillTraining, self).__init__(
      net, train_data, validate_data,
      optimizer=optimizer,
      schedule=schedule,
      max_epochs=max_epochs,
      device=device,
      checkpoint_path=checkpoint_path
    )
    self.relabel_data = relabel_data
    self.classes = len(relabel_data)
    self.label_spill_matrix = torch.eye(self.classes)
    self.relabel_steps = relabel_steps

  def relabel(self):
    for idx, data in enumerate(iter(self.relabel_data)):
      input, _ = data.values()
      input = input.to(self.device)
      predictions = self.net(input).sum(dim=0)
      predictions /= input.size()[0]
      self.label_spill_matrix[idx, :] = predictions[:].squeeze()

  def each_relabel(self):
    pass

  def validate(self):
    vit = iter(self.validate_data)
    vinputs, vlabels = next(vit).values()
    vinputs, vlabels = vinputs.to(device), vlabels.to(device)
    if self.relabel_net != None:
      vlabels = self.label_spill_matrix.dot(vlabels)
    voutputs = self.net(vinputs)
    vloss_val = self.loss(voutputs, vlabels)
    self.validation_loss = vloss_val.item()

  def train(self):
    for epoch_id in range(max_epochs):
      self.epoch_id = epoch_id
      validation_loss = None
      for data in self.train_data:
        input, label = data.values()
        if self.relabel_net != None:
          label = self.label_spill_matrix.dot(label)
        self.step(input, label)
        self.validate()
        self.each_step()
        self.step_id += 1
      if (self.epoch_id + 1) % self.relabel_steps:
        self.relabel()
        self.each_relabel()
      self.schedule_step()
      self.each_epoch()