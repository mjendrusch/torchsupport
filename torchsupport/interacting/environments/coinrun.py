import torch
import gym

from torchsupport.interacting.environments.environment import Environment

class CoinRun(Environment):
  def __init__(self, history=1):
    self.env = gym.make("procgen:procgen-coinrun-v0", use_sequential_levels=True)
    self.initialized = False
    self.state = None
    self.done = False
    self.history = history
    self._history = torch.zeros(self.history, 3, 64, 64)

  def add_history(self, state):
    self._history = self._history.roll(1, dims=0)
    self._history[0] = state
    return self._history.view(self.history * 3, 64, 64)

  def reset(self):
    self._history.zero_()
    state = torch.tensor(self.env.reset(), dtype=torch.float)
    state = state.permute(2, 0, 1).contiguous() / 255
    self.state = self.add_history(state)
    self.initialized = True
    self.done = False

  def act(self, action):
    observation, reward, done, _ = self.env.step(int(action))
    state = torch.tensor(observation, dtype=torch.float) / 255
    state = state.permute(2, 0, 1).contiguous()
    self.state = self.add_history(state)
    self.done = done
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
    state = torch.zeros(self.history * 3, 64, 64)
    reward = torch.tensor(0.0)
    done = torch.tensor(0)
    action = torch.tensor(0)
    sample = self.data_type(
      state=state, action=action, rewards=reward, done=done
    )
    return sample
