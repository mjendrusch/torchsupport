import torch
import torch.nn as nn
import torch.nn.functional as func

class ReZero(nn.Module):
  r'''Implemets ReZero normalization proposed by Bachlechner et al. 
  (https://arxiv.org/pdf/2003.04887.pdf) 
  Args: 
    function (callable): function you want to use (for example conv2D)
    out_size (int): dimension of the channel output'''
  def __init__(self, function, out_size=1):
    super().__init__()
    self.out_size = out_size
    self.function = function
    self.alpha = nn.Parameter(torch.zeros(
      self.out_size, dtype=torch.float, requires_grad=True
    ))

  def forward(self, inputs):
    dimension = inputs.dim() - 2
    alpha = self.alpha[[None, slice(None)] + dimension * [None]]
    return inputs + alpha * self.function(inputs)

