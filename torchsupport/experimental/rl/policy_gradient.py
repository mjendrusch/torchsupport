import torch
import torch.nn
import torch.nn.functional as func

class MonteCarloPolicyGradient(nn.Module):
  def __init__(self, reward, discount=None, baseline=False, decay=0.1):
    """Monte Carlo policy gradient loss (REINFORCE).

    Args:
      reward (callable): computes the reward of a `Trajectory`.
      discount (int or None): reward discount.
      baseline (bool): use baseline?
      decay (float): baseline decay.
    """
    super(MonteCarloPolicyGradient, self).__init__()
    self.reward = reward
    self.discount = discount
    self.baseline = None
    self.baseline_decay = decay
    if baseline:
      self.baseline = torch.tensor(0.0)

  def forward(self, trajectory):
    raw_rewards = self.reward(trajectory)
    if self.discount != None:
      rewards = torch.zeros_like(raw_rewards)
      R = 0
      for idx, r in zip(range(raw_rewards.size(0), -1, -1), raw_rewards[::-1]):
        R = r + R * self.discount
        rewards[idx] = R
    else:
      rewards = raw_rewards

    if self.baseline != None:
      advantage = rewards - self.baseline
      self.baseline = self.reward + self.baseline_decay * (self.baseline - self.reward)
    else:
      advantage = rewards

    log_probability = torch.log(trajectory.probabilities[trajectory.actions])
    loss = -(log_probability * advantage).sum()
    
    return loss

class AdvantageActorCritic(MonteCarloPolicyGradient):
  def __init__(self, reward, critic, discount=0.1, decay=0.1):
    """Advantage Actor Critic loss (A2C).

    Args:
      reward (callable): computes the reward of a `Trajectory`.
      critic (callable): approximates the values in a `Trajectory`.
      discount (int or None): reward discount.
      decay (float): baseline decay.
    """
    super(AdvantageActorCritic, self).__init__(reward, discount=discount, decay=decay)
    assert discount != None
    self.critic = critic

  def forward(self, trajectory):
    value = self.critic(trajectory.states)
    raw_rewards = self.reward(trajectory)
    rewards = torch.zeros_like(raw_rewards)
    R = 0 if trajectory.state_is_terminal(-1) else value[-1]
    for idx, r in zip(range(raw_rewards.size(0), -1, -1))
      R = r + R * self.discount
      rewards[idx] = R

    advantage = rewards - value[:-1]
    log_probability = torch.log(trajectory.probabilities[trajectory.actions])

    critic_loss = ((value - rewards) ** 2).mean()
    actor_loss = -(log_probability * advantage).mean()

    return actor_loss + critic_loss

class ProximalPolicyGradient(AdvantageActorCritic):
  def __init__(self, reward, critic, discount=0.1, decay=0.1, clip=0.2):
    """Proximal Policy Optimization loss (PPO).

    Args:
      reward (callable): computes the reward of a `Trajectory`.
      critic (callable): approximates the values in a `Trajectory`.
      discount (int or None): reward discount.
      decay (float): baseline decay.
      clip (float): surrogate clipping parameter.
    """
    super(ProximalPolicyGradient, self).__init__(reward, critic, discount=discount, decay=decay)
    assert discount != None
    self.clip = clip
    self.old_policy = None

  def forward(self, trajectory):
    old_probabilities = []
    for state in trajectory.states:
      old_probabilities.append(self.old_policy(state))

    R = 0 if trajectory.state_is_terminal(-1) else value[-1]
    for idx, r in zip(range(raw_rewards.size(0), -1, -1))
      R = r + R * self.discount
      rewards[idx] = R

    advantage = rewards - value[:-1]
    ratio = trajectory.probabilities[trajectory.actions] / old_probabilities
    s1 = ratio * advantage
    s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage
    
    critic_loss = ((value - rewards) ** 2).mean()
    actor_loss = -torch.min(s1, s2).mean()

    return actor_loss + critic_loss
