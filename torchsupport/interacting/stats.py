import torch
import torch.multiprocessing as mp

from torchsupport.interacting.control import ReadWriteControl

class Statistics:
  def update(self, stats):
    pass

class ExperienceStatistics(Statistics):
  def __init__(self, decay=0.6):
    self.ctrl = ReadWriteControl(self)
    self.decay = decay
    self._total = torch.tensor([0.0])
    self._length = torch.tensor([0.0])
    self._total_steps = torch.tensor([0])
    self._total.share_memory_()
    self._length.share_memory_()
    self._total_steps.share_memory_()

  def pull_changes(self):
    pass

  def push_changes(self):
    pass

  def update(self, stats):
    with self.ctrl.write:
      self._total[0] = (1 - self.decay) * stats.total + self.decay * self._total[0]
      self._length[0] = (1 - self.decay) * stats.length + self.decay * self._length[0]
      self._total_steps[0] = self._total_steps[0] + stats.length

  @property
  def total(self):
    with self.ctrl.read:
      return self._total

  @property
  def length(self):
    with self.ctrl.read:
      return self._length

  @property
  def steps(self):
    with self.ctrl.read:
      return self._total_steps

class EnergyStatistics(Statistics):
  pass
