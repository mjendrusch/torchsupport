import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.distributed as distributed

from torchsupport.data.io import netwrite
from torchsupport.training.training import Training, BasicTraining

class SynchronousDistributedTraining(BasicTraining):
  """Distributes a given training process over a set of nodes,
  via gradient averaging.
  """
  def __init__(self, *args, **kwargs):
    super(SynchronousDistributedTraining, self).__init__(*args, **kwargs)
    self.world_size = distributed.get_world_size()
    self.rank = distributed.get_rank()
    self.group = distributed.new_group(ranks=list(range(self.world_size)))

  def step(self, data, label):
    predictions = self.net(data)
    loss_val = self.loss(predictions, label)
    loss_val.backwards()
    _average_gradients(self.net, self.world_size, self.group)
    self.optimizer.step()
    self.training_loss = loss_val.item()
  
  def checkpoint(self):
    if self.rank == 0:
      super(SynchronousDistributedTraining, self).checkpoint()

class AsynchronousDistributedTraining(BasicTraining):
  """Distribute a given training process over a set of nodes,
  via GossipGraD distributed training.
  """
  def __init__(self, *args, **kwargs):
    super(AsynchronousDistributedTraining, self).__init__(*args, **kwargs)
    self.gossip_step = 0
    self.world_size = distributed.get_world_size()
    self.rank = distributed.get_rank()
    self.groups = []
    for idx in range(self.world_size - 1):
      partner = (self.rank + idx + 1) % self.world_size
      group = distributed.new_group(ranks=[self.rank, partner])
      self.groups.append(group)

  def step(self, data, label):
    predictions = self.net(data)
    loss_val = self.loss(predictions, label)
    loss_val.backwards()
    _gossip_grad(self.net, self.world_size, self.rank,
                 self.groups, self.gossip_step)
    self.gossip_step += 1
    if self.gossip_step == self.world_size - 1:
      self.gossip_step = 0
    self.optimizer.step()
    self.training_loss = loss_val.item()

  def checkpoint(self):
    if self.rank == 0:
      super(AsynchronousDistributedTraining, self).checkpoint()

def _average_gradients(net, world_size, group, cuda=False):
  for p in net.parameters():
    tensor = p.grad.data.cpu()
    distributed.all_reduce(tensor,
                           op=distributed.reduce_op.SUM,
                           group=group)
    tensor /= float(world_size)
    if cuda:
      p.grad.data = tensor.cuda()
    else:
      p.grad.data = tensor

def _gossip_grad(net, world_size, rank, groups, step, cuda=False):
  group = groups[step]
  for p in net.parameters():
    tensor = p.grad.data.cpu()
    distributed.all_reduce(tensor,
                           op=distributed.reduce_op.SUM,
                           group=group)
    tensor /= 2.0
    if cuda:
      p.grad.data = tensor.cuda()
    else:
      p.grad.data = tensor
