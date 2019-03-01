import random
from copy import copy
import torch
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from torchsupport.data.io import netwrite
from torchsupport.data.episodic import SupportData

class Training(object):
  """Abstract training process class.
  """
  def __init__(self):
    pass

  def each_step(self):
    pass

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
  def __init__(self, net, train_data, validate_data, losses,
               optimizer=torch.optim.Adam,
               schedule=None,
               max_epochs=50,
               batch_size=128,
               device="cpu",
               network_name="network",
               path_prefix=".",
               valid_callback=lambda x: None):
    super(SupervisedTraining, self).__init__()
    self.valid_callback = valid_callback
    self.network_name = network_name
    self.writer = SummaryWriter(network_name)
    self.device = device
    self.optimizer = optimizer(net.parameters())
    if schedule is None:
      self.schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)
    else:
      self.schedule = schedule
    self.losses = losses
    self.train_data = DataLoader(
      train_data, batch_size=batch_size, num_workers=8, shuffle=True
    )
    self.validate_data = DataLoader(
      validate_data, batch_size=8 * batch_size, shuffle=False
    )
    self.net = net.to(self.device)
    self.max_epochs = max_epochs
    self.checkpoint_path = f"{path_prefix}/{network_name}-checkpoint"
    self.step_id = 0
    self.epoch_id = 0
    self.validation_losses = [0 for _ in range(len(self.losses))]
    self.training_losses = [0 for _ in range(len(self.losses))]
    self.best = None

  def checkpoint(self):
    netwrite(
      self.net,
      f"{self.checkpoint_path}-epoch-{self.epoch_id}-step-{self.step_id}.torch"
    )
    self.each_checkpoint()

  def step(self, data, label):
    self.optimizer.zero_grad()
    predictions = self.net(data)

    loss_val = torch.tensor(0.0).to(self.device)
    if isinstance(predictions, (list, tuple)):
      for idx, prediction in enumerate(predictions):
        this_loss_val = self.losses[idx](prediction, label[idx])
        self.training_losses[idx] = float(this_loss_val)
        loss_val += this_loss_val
    else:
      this_loss_val = self.losses[0](predictions, label[0])
      self.training_losses[0] = float(this_loss_val)
      loss_val += this_loss_val
    loss_val.backward()
    self.optimizer.step()
    self.each_step()

  def validate(self):
    with torch.no_grad():
      vit = iter(self.validate_data)
      inputs, *label = next(vit)
      inputs, label = inputs.to(self.device), list(map(lambda x: x.to(self.device), label))
      predictions = self.net(inputs)
      if isinstance(predictions, (list, tuple)):
        for idx, prediction in enumerate(predictions):
          this_loss_val = self.losses[idx](prediction, label[idx])
          self.validation_losses[idx] = float(this_loss_val)
      else:
        self.validation_losses[0] = self.losses[0](predictions, label[0])
      self.each_validate()
      self.valid_callback(self, inputs.to("cpu").numpy(), list(map(lambda x: x.to("cpu"), label)))

  def schedule_step(self):
    self.schedule.step(sum(self.validation_losses))

  def each_step(self):
    for idx, loss in enumerate(self.training_losses):
      self.writer.add_scalar(f"training loss {idx}", loss, self.step_id)
    self.writer.add_scalar(f"training loss total", sum(self.training_losses), self.step_id)

  def each_validate(self):
    for idx, loss in enumerate(self.validation_losses):
      self.writer.add_scalar(f"validation loss {idx}", loss, self.step_id)
    self.writer.add_scalar(f"validation loss total", sum(self.validation_losses), self.step_id)
    if self.step_id % 50 == 0:#self.best is None or sum(self.validation_losses) < self.best:
      self.best = sum(self.validation_losses)
      self.checkpoint()

  def train(self):
    for epoch_id in range(self.max_epochs):
      self.epoch_id = epoch_id
      for data in self.train_data:
        inputs, *label = data
        inputs, label = inputs.to(self.device), list(map(lambda x: x.to(self.device), label))
        self.step(inputs, label)
        if self.step_id % 10 == 0:
          self.validate()
        self.step_id += 1
      # self.schedule_step() # FIXME
      self.each_epoch()
    return self.net

class FewShotTraining(SupervisedTraining):
  def __init__(self, net, train_data, validate_data, losses,
               optimizer=torch.optim.Adam,
               schedule=None,
               max_epochs=50,
               batch_size=128,
               device="cpu",
               network_name="network",
               path_prefix=".",
               valid_callback=lambda x: None):
    super(FewShotTraining, self).__init__(
      net, train_data, validate_data, losses,
      optimizer=optimizer,
      schedule=schedule,
      max_epochs=max_epochs,
      batch_size=batch_size,
      device=device,
      network_name=network_name,
      path_prefix=path_prefix,
      valid_callback=valid_callback
    )

    support_data = copy(train_data)
    train_data.data_mode = type(train_data.data_mode)(1)
    support_data = SupportData(train_data, shots=5)
    validate_support_data = SupportData(validate_data, shots=5)
    self.support_loader = iter(DataLoader(support_data))
    self.valid_support_loader = iter(DataLoader(validate_support_data))

  def step(self, data, label):
    self.optimizer.zero_grad()

    permutation = [0, 1, 2]
    random.shuffle(permutation)

    support, support_label = next(self.support_loader)

    lv = label[0].reshape(-1)
    for idx, val in enumerate(lv):
      lv[idx] = permutation[int(val[0])]
    lv = support_label.reshape(-1)
    print(lv.size())
    for idx, val in enumerate(lv):
      print(permutation[int(val[0])], permutation, val)
      lv[idx] = permutation[int(val[0])]

    support = support[0].to(self.device)
    support_label = support_label[0].to(self.device)
    predictions = self.net(data, support, support_label)

    loss_val = torch.tensor(0.0).to(self.device)
    if isinstance(predictions, (list, tuple)):
      for idx, prediction in enumerate(predictions):
        this_loss_val = self.losses[idx](prediction, label[idx])
        self.training_losses[idx] = float(this_loss_val)
        loss_val += this_loss_val
    else:
      print(predictions[:5])
      this_loss_val = self.losses[0](predictions, label[0])
      self.training_losses[0] = float(this_loss_val)
      loss_val += this_loss_val
    loss_val.backward()
    self.optimizer.step()
    self.each_step()  

  def validate(self):
    with torch.no_grad():
      self.net.eval()
      vit = iter(self.validate_data)
      inputs, *label = next(vit)
      inputs, label = inputs.to(self.device), list(map(lambda x: x.to(self.device), label))
      support, support_label = next(self.valid_support_loader)
      support = support[0].to(self.device)
      support_label = support_label[0].to(self.device)
      predictions = self.net(inputs, support, support_label)
      if isinstance(predictions, (list, tuple)):
        for idx, prediction in enumerate(predictions):
          this_loss_val = self.losses[idx](prediction, label[idx])
          self.validation_losses[idx] = float(this_loss_val)
      else:
        self.validation_losses[0] = self.losses[0](predictions, label[0])
      self.each_validate()
      self.valid_callback(self, inputs.to("cpu").numpy(), list(map(lambda x: x.to("cpu"), label)))
      self.net.train()
