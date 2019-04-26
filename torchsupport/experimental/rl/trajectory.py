import torch
import torch.nn

class Trajectory(object):
  def __init__(self, states=[], probabilities=[], choices=[]):
    """Reinforcement learning agent trajectory.

    Args:
      states (list): trajectory states.
      probabilities (list): trajectory probabilities.
      choices (list): trajectory choices.
    """
    self._states = states
    self._probabilities = probabilities
    self._choices = choices

  def __len__(self):
    return len(self._states)

  def __getitem__(self, idx):
    if isinstance(idx, slice):
      return Trajectory(
        states=self._states[idx],
        probabilities=self._probabilities[idx],
        choices=self._choices[idx]
      )
    else:
      return (
        self._states[idx],
        self._probabilities[idx],
        self._choices[idx]
      )

  @property
  def states(self):
    return self._states

  @property
  def probabilities(self):
    prob = []
    for idx, choice in enumerate(self._choices):
      prob.append(self._probabilities[idx][choice])
    return torch.cat(prob)

  @property
  def choices(self):
    return torch.cat(self._choices)

  def append(self, state, probability, choice):
    self._states.append(state)
    self._probabilities.append(probability)
    self._choices.append(choice)
