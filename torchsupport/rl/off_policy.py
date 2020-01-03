import time

import torch

from torchsupport.data.io import to_device, netwrite

from torchsupport.training.training import Training

from torchsupport.rl.trajectory import Trajectory
from torchsupport.rl.sampler import Sampler
from torchsupport.rl.data import ExperienceGenerator, ReplayBuffer

class OffPolicyTraining(Training):
  def __init__(self, agent, environment,
               optimizer=torch.optim.Adam,
               optimizer_kwargs=None,
               max_steps=1000,
               batch_size=128,
               trajectory_length=0,
               trajectory_kind=Trajectory,
               trajectory_kwargs=None,
               sampler_kind=Sampler,
               sampler_kwargs=None,
               generator_kind=ExperienceGenerator,
               generator_kwargs=None,
               replay_kind=ReplayBuffer,
               replay_kwargs=None,
               device="cpu",
               network_name="network",
               path_prefix=".",
               report_interval=10,
               checkpoint_interval=1000):
    super().__init__()
    optimizer_kwargs = optimizer_kwargs or {}
    trajectory_kwargs = trajectory_kwargs or {}
    sampler_kwargs = sampler_kwargs or {}
    generator_kwargs = generator_kwargs or {}
    replay_kwargs = replay_kwargs or {}
    self.device = device
    self.network_name = network_name
    self.path_prefix = path_prefix
    self.path = f"{self.path_prefix}/{self.network_name}"
    self.report_interval = report_interval
    self.checkpoint_interval = checkpoint_interval

    self.agent = agent
    self.environment = environment
    self.optimizer = optimizer(
      self.agent.parameters(),
      **optimizer_kwargs
    )

    self.max_steps = max_steps
    self.batch_size = batch_size
    self.trajectory_length = trajectory_length

    self.generator = generator_kind(
      agent, environment,
      sampler_kind=sampler_kind,
      trajectory_kind=trajectory_kind
    )

    self.replay = replay_kind(**replay_kwargs)

    self.training_losses = {}

    self.step_id = 0
    self.epoch_id = 0

  def save_path(self):
    return f"{self.path}-save.torch"

  def each_checkpoint(self):
    netwrite(self.agent, f"{self.path}-checkpoint-{self.step_id}.torch")

  def loss(self, *args, **kwargs):
    raise NotImplementedError("Abstract.")

  def run_networks(self, data):
    return self.agent(data)

  def update(self, *outputs):
    self.optimizer.zero_grad()
    loss_val = self.loss(*outputs)
    loss_val.backward()
    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 5.0)
    self.optimizer.step()

  def step(self, data):
    outputs = self.run_networks(data)
    self.update(*outputs)
    self.each_step()

  def train(self):
    self.generator.start(self.agent)

    while len(self.replay) < 2 * self.batch_size:
      self.replay.append(self.generator.get_experience())
      time.sleep(1)

    for step in range(self.max_steps):
      batch = self.replay.get_batch(self.batch_size, self.trajectory_length)
      batch = to_device(batch, self.device)
      self.step(batch)
      if self.step_id % self.report_interval == 0:
        self.each_validate()
      if self.step_id % self.checkpoint_interval == 0:
        self.each_checkpoint()
      self.replay.append(self.generator.get_experience())
      self.generator.update_model(self.agent)
      self.save_tick()
      self.step_id += 1

    self.generator.join()
