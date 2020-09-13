import torch
import torch.nn as nn
import torch.nn.functional as func

class ReZero(nn.Module):
  r'''Implemets ReZero normalization proposed by Bachlechner et al. 
    (https://arxiv.org/pdf/2003.04887.pdf) 
  Args: 
    function (callable): function you want to use (for example conv2D)'''
  def __init__(self, function):
    super().__init__()
    self.function = function
    self.alpha = nn.Parameter(torch.tensor(
      0, dtype=torch.float, requires_grad=True
    ))

  def forward(self, inputs):
    return inputs + self.alpha * self.function(inputs)
