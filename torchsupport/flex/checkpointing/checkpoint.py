import os
import random
import numpy as np
import torch
from torchsupport.data.io import netread, netwrite
from torchsupport.flex.checkpointing.savable import Savable, SaveStateError

class Checkpoint:
  def __init__(self, ctx):
    self.ctx = ctx
    self.checkpoint_names = {}
    self.save_names = {"context": ctx}

  def add_checkpoint(self, **kwargs):
    self.checkpoint_names.update(kwargs)
    self.save_names.update(kwargs)

  def add_save(self, **kwargs):
    self.save_names.update(kwargs)

  def save_path(self):
    return f"{self.ctx.path}-save.torch"

  def checkpoint(self):
    for name, the_net in self.checkpoint_names.items():
      if isinstance(the_net, torch.nn.DataParallel):
        the_net = the_net.module
      netwrite(
        the_net,
        f"{self.ctx.path}-{name}-step-{self.ctx.step_id}.torch"
      )

  def emergency_read_checkpoint(self):
    import glob
    for name, the_net in self.checkpoint_names.items():
      if isinstance(the_net, torch.nn.DataParallel):
        the_net = the_net.module
      files = glob.glob(f"{self.ctx.path}-{name}-epoch-*.torch")
      files = sorted(files, key=lambda x: int(x.split("-")[-1].split(".")[0]))
      target = files[-1]
      netread(
        the_net,
        target
      )

  def write(self, path):
    data = {}
    data["_torch_rng_state"] = torch.random.get_rng_state()
    data["_np_rng_state"] = np.random.get_state()
    data["_random_rng_state"] = random.getstate()
    for name, param in self.save_names.items():
      param = Savable.wrap(param)
      param.write(data, name)
    torch.save(data, path + ".tmp")
    if os.path.isfile(path):
      os.rename(path, path + ".old")
    os.rename(path + ".tmp", path)

  def read(self, path):
    data = torch.load(path)
    torch.random.set_rng_state(data["_torch_rng_state"])
    np.random.set_state(data["_np_rng_state"])
    random.setstate(data["_random_rng_state"])
    for name, param in self.save_names.items():
      param = Savable.wrap(param)
      param.read(data, name)

  def save(self, path=None):
    path = path or self.save_path()
    try:
      self.write(path)
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
