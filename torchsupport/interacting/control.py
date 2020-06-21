from copy import copy
import torch.multiprocessing as mp

class ReadControl:
  def __init__(self, ctrl):
    self.ctrl = ctrl

  def __enter__(self):
    with self.ctrl.read_lock:
      self.ctrl.read_count.value += 1
      if self.ctrl.read_count.value == 1:
        self.ctrl.write_lock.acquire()
    self.ctrl.owner.pull_changes()

  def __exit__(self, *args):
    with self.ctrl.read_lock:
      self.ctrl.read_count.value -= 1
      if self.ctrl.read_count.value == 0:
        self.ctrl.write_lock.release()

class WriteControl:
  def __init__(self, ctrl):
    self.ctrl = ctrl

  def __enter__(self):
    self.ctrl.write_lock.acquire()
    self.ctrl.owner.pull_changes()

  def __exit__(self, *args):
    self.ctrl.owner.push_changes()
    self.ctrl.write_lock.release()

class ReadWriteControl:
  def __init__(self, owner):
    self.owner = owner
    self.read_lock = mp.Lock()
    self.write_lock = mp.Lock()
    self.read_count = mp.Value("l", 0)
    self.read_count.value = 0

    self.timestamp = mp.Value("l", 0)
    self.local_timestamp = 0

  def clone(self, owner):
    result = copy(self)
    result.owner = owner
    return result

  def change(self, toggle=True):
    self.timestamp.value = self.timestamp.value + 1
    self.local_timestamp = self.timestamp.value

  def advance(self):
    self.local_timestamp = self.timestamp.value

  @property
  def changed(self):
    timestamp = self.timestamp.value
    local_timestamp = self.local_timestamp
    return local_timestamp != timestamp

  @property
  def read(self):
    return ReadControl(self)

  @property
  def write(self):
    return WriteControl(self)
