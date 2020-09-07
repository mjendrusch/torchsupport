from abc import ABC, abstractmethod

import torch
import torch.multiprocessing as mp

from torchsupport.rl.trajectory import Experience, Trajectory
from torchsupport.rl.data import Flag, SharedModel, SharedModelList

class Task(ABC):
  """Integrated task definition for multi-task imitation learning."""
  def __init__(self, policies, env):
    self.policies = policies
    self.environment = env

  @abstractmethod
  def define(self):
    """Defines a task parameters for sampling."""
    return None

  @abstractmethod
  def run_policies(self, initial_state, inputs=None):
    """Runs policies for a single sampling step."""
    return None

  @abstractmethod
  def sample(self, agent_output, agent_state):
    """Samples from policy output for a single step."""
    return None

  @abstractmethod
  def distill(self):
    """Distills information from a sampled trajectory."""
    return None

  def step(self, inputs=None):
    """Performs a sinlge step in the environment."""
    initial_state = self.environment.observe()

    agent_output, agent_state = self.run_policies(initial_state, inputs)

    action = self.sample(agent_output, agent_state)
    reward = self.environment.act(action)
    terminal = int(self.environment.is_done())
    final_state = self.environment.observe()

    return Experience(
      initial_state, final_state, action, reward,
      terminal=terminal, logits=agent_output, outputs=agent_state
    )

  def sample_episode(self, kind=Trajectory):
    """Samples an episode from the environment."""
    with torch.no_grad():
      self.environment.reset()
      trajectory = kind()
      inputs = None
      while not self.environment.is_done():
        experience = self.step(inputs=inputs)
        inputs = experience.outputs
        trajectory.append(experience)
      trajectory.complete()
      return trajectory

  @property
  def contents(self):
    return self.policies, self.environment

class TaskGenerator:
  def __init__(self, contents, n_workers=4, max_items=64, kind=Task):
    self.kind = kind
    self.policies, self.env = contents
    self.shared_policies = SharedModelList(self.policies)
    self.shared_env = self.env
    self.done = Flag(0)
    self.max_items = max_items
    self.n_workers = n_workers
    self.queue = mp.Queue(maxsize=2 * max_items)
    self.procs = []

  @property
  def worker(self):
    """Returns a runnable worker."""
    def _task_worker(queue, shared_contents, contents, done, kind):
      models, environment = contents
      shared_models, shared_environment = shared_contents
      sampler = kind.from_contents(contents)
      while True:
        if done.read():
          break
        shared_models.read(models)
        result = sampler.sample_episode(kind=Trajectory)
        queue.put(result)
    return _task_worker

  def start(self):
    for idx in range(self.n_workers):
      proc = mp.Process(
        target=self.worker,
        args=(
          self.queue,
          (self.shared_policies, self.shared_env),
          (self.policies, self.env),
          self.done,
          self.kind)
      )
      self.procs.append(proc)
      proc.start()

  def join(self):
    self.done.update(1)
    while not self.queue.empty():
      _ = self.queue.get_nowait()
    for proc in self.procs:
      proc.join()
    self.procs = []

  def update_model(self, policies):
    self.shared_policies.update(policies)

  def update_env(self, parameters):
    pass # TODO

  def get_experience(self):
    results = []
    got = 0
    while not self.queue.empty():
      if got >= self.max_items:
        break
      results.append(self.queue.get().clone())
      got += 1
    return results
