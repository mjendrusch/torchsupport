import random
from copy import deepcopy

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset

from torchsupport.data.collate import default_collate
from torchsupport.rl.sampler import Sampler
from torchsupport.rl.trajectory import Trajectory

class Flag:
  def __init__(self, data):
    self.lock = mp.Lock()
    self.data = mp.Value("i", data)

  def update(self, data):
    with self.lock:
      self.data.value = data

  def read(self):
    with self.lock:
      return self.data.value

class SharedModel:
  def __init__(self, model):
    self.lock = mp.Lock()
    self.model = model.share_memory()
    self.state_dict = model.state_dict()

  def update(self, model):
    with self.lock:
      self.model.load_state_dict(model.state_dict())

  def read(self, model=None):
    if model is None:
      model = deepcopy(self.model)
    with self.lock:
      model.load_state_dict(self.model.state_dict())
    return model

class SharedModelList:
  def __init__(self, models):
    self.models = [
      SharedModel(model)
      for model in models
    ]

  def update(self, models):
    for model, reference in zip(models, self.models):
      reference.update(model)

  def read(self, models=None):
    if models is None:
      models = [
        deepcopy(model.model)
        for model in self.models
      ]
    for model, reference in zip(models, self.models):
      reference.read(model=model)
    return models

class ExperienceGenerator:
  def __init__(self, model, env,
               sampler_kind=Sampler,
               trajectory_kind=Trajectory,
               n_workers=4,
               max_items=64):
    self.n_workers = n_workers
    self.model = SharedModel(deepcopy(model))
    self.env = env
    self.done = Flag(0)
    self.max_items = max_items
    self.queue = mp.Queue(maxsize=2 * max_items)
    self.procs = []
    self.sampler_kind = sampler_kind
    self.trajectory_kind = trajectory_kind

  def start(self, model):
    for idx in range(self.n_workers):
      proc = mp.Process(
        target=_sampler_worker,
        args=(
          self.queue, self.model, model, self.env, self.done,
          self.sampler_kind, self.trajectory_kind)
      )
      self.procs.append(proc)
      proc.start()

  def join(self):
    self.done.update(1)
    while not self.queue.empty():
      print("rolling here")
      _ = self.queue.get_nowait()
    for proc in self.procs:
      print("rolling there")
      proc.join()
    self.procs = []

  def update_model(self, model):
    self.model.update(model)

  def update_env(self, parameters):
    pass # TODO

  def get_experience(self):
    results = []
    got = 0
    while not self.queue.empty():
      if got >= self.max_items:
        break
      retrieved = self.queue.get()
      results.append(retrieved.torch())
      del retrieved
      got += 1
    try:
      print("length", sum(map(len, results)) / len(results))
    except:
      pass
    return results

def _sampler_worker(queue, shared_model, model, environment, done, kind, traj):
  sampler = kind(model, environment)
  torch.set_num_threads(1)
  while True:
    if done.read():
      print("DONE")
      break
    model = shared_model.read(model=model)
    result = sampler.sample_episode(kind=traj)
    queue.put(result.numpy())

class ReplayBuffer(Dataset):
  def __init__(self, size=10000):
    self.size = size
    self.trajectories = []

  def append(self, trajectories):
    self.trajectories += trajectories
    if len(self.trajectories) > self.size:
      del self.trajectories[:len(self.trajectories) - self.size]

  def __getitem__(self, index):
    return self.trajectories[index]

  def get_batch(self, size=64, length=0):
    items = []
    for idx in range(size):
      index = random.randrange(0, len(self))
      traj = self[index]
      offset = random.randrange(0, len(traj) - length)
      window = ...
      if length == 0:
        window = traj[offset]
      else:
        window = traj[offset:offset+length]
      items.append(window)
    return default_collate(items)

  def __len__(self):
    return len(self.trajectories)
