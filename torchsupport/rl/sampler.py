import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.rl.trajectory import Trajectory, Experience

class Sampler:
  def __init__(self, agent, environment):
    self.agent = agent
    self.trajectories = []
    self.environment = environment

  def step(self, inputs=None):
    initial_state = self.environment.observe()
    logits, output = self.agent(
      initial_state.unsqueeze(0),
      inputs
    )
    logits = logits[0]
    output = output if output is None else output[0]
    action = self.agent.sample(logits)
    reward = self.environment.act(action)
    terminal = int(self.environment.is_done())
    final_state = self.environment.observe()

    return Experience(
      initial_state, final_state, action, reward,
      terminal=terminal, logits=logits, outputs=output
    )

  def sample_episode(self, kind=Trajectory):
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
