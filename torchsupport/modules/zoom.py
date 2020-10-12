import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.gradient import hard_k_hot

def zoom2d(target, logits, offset=64, num_samples=2, temperature=0.1):
  shape = logits.shape[2:]
  logits = logits.view(logits.size(0), -1)
  k_hot = hard_k_hot(logits, num_samples, temperature=temperature)
  k_hot = k_hot.view(logits.size(0), *shape)
  ind, x, y = k_hot.nonzero().t()

  x_ind = torch.arange(0, offset)[None]
  x_ind = x_ind.repeat_interleave(num_samples * logits.size(0), dim=0)
  x_ind = x_ind + x[:, None]

  y_ind = torch.arange(0, offset)[None]
  y_ind = y_ind.repeat_interleave(num_samples * logits.size(0), dim=0)
  y_ind = y_ind + y[:, None]

  result = target[ind[:, None, None], :, x_ind[:, :, None], y_ind[:, None, :]]
  result = result.unsqueeze(1).transpose(1, -1).squeeze(-1)
  result = result * k_hot[ind, x, y][:, None, None, None]
  result = result.reshape(-1, num_samples, *result.shape[1:])
  return result

class Zoom2d(nn.Module):
  def __init__(self, offset=64, num_samples=2, temperature=0.1):
    super().__init__()
    self.offset = offset
    self.num_samples = num_samples
    self.temperature = temperature

  def forward(self, target, logits):
    return zoom2d(
      target, logits, offset=self.offset,
      num_samples=self.num_samples,
      temperature=self.temperature
    )
