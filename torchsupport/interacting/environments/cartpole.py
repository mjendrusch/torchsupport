import torch
import gym

from torchsupport.interacting.environments.environment import Environment

class CartPole(Environment):
  def __init__(self, scale=False):
    self.env = gym.make("CartPole-v1")
    self.initialized = False
    self.state = None
    self.done = False
    self.scale = scale

  def reset(self):
    self.state = torch.tensor(self.env.reset(), dtype=torch.float)
    self.initialized = True
    self.done = False

  def act(self, action):
    observation, reward, done, _ = self.env.step(int(action))
    self.state = torch.tensor(observation, dtype=torch.float)
    self.done = done
    if self.scale:
      reward = reward / 100
    return torch.tensor([reward])

  def observe(self):
    return self.state

  def is_done(self):
    return self.done

  @property
  def action_space(self):
    return self.env.action_space

  @property
  def observation_space(self):
    return self.env.observation_space

  def schema(self):
    state = torch.tensor([0.0] * 4)
    reward = torch.tensor(0.0)
    done = torch.tensor(0)
    action = torch.tensor(0)
    sample = self.data_type(
      state=state, action=action, rewards=reward, done=done
    )
    return sample
