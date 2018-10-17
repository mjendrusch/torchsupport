import torch
import torch.nn as nn
import torch.nn.functional as func

class ShortCutSender(nn.Module):
  def __init__(self, sender):
    super(ShortCutSender, self).__init__()
    self.sender = sender
    self.result = None

  def forward(self, input):
    self.result = self.sender(input)
    return self.result

class ShortCutReceiver(nn.Module):
  def __init__(self, receiver, sender, combination):
    super(ShortCutReceiver, self).__init__()
    self.receiver = receiver
    self.sender = sender
    self.combination = combination

  def forward(self, input):
    shortcut_input = self.combination(self.sender.result, input)
    return self.receiver(shortcut_input)

def shortcut(sender, receiver, combination=lambda x, y: torch.cat([x, y], dim=1)):
  """
  usage:
  sender, receiver = shortcut(sender, receiver, combination)
  """
  shortcut_sender = ShortCutSender(sender)
  shortcut_receiver = ShortCutReceiver(receiver, sender, combination)
  return shortcut_sender, shortcut_receiver

def down_up_pair(size, stride):
  pass
