import torch
import torch.nn as nn
import torch.nn.functional as func

class ConvGRUCellNd(nn.Module):
  def __init__(self, in_size, out_size, kernel_size, N=1, **kwargs):
    super(ConvGRUCellNd, self).__init__()
    conv = eval(f"nn.Conv{N}d")
    self.conv_ir = conv(in_size, out_size, kernel_size, **kwargs)
    self.conv_hr = conv(in_size, out_size, kernel_size, **kwargs)
    self.conv_iz = conv(in_size, out_size, kernel_size, **kwargs)
    self.conv_hz = conv(in_size, out_size, kernel_size, **kwargs)
    self.conv_in = conv(in_size, out_size, kernel_size, **kwargs)
    self.conv_hn = conv(in_size, out_size, kernel_size, **kwargs)

  def forward(self, inputs, state):
    r = torch.sigmoid(self.conv_ir(inputs) + self.conv_hr(state))
    z = torch.sigmoid(self.conv_iz(inputs) + self.conv_hz(state))
    n = torch.tanh(self.conv_in(inputs) + self.conv_hn(state * r))
    return z * state + (1 - z) * n

class ConvGRUCell1d(ConvGRUCellNd):
  def __init__(self, in_size, out_size, hidden_size, kernel_size, **kwargs):
    super().__init__(in_size, out_size, hidden_size, kernel_size, N=1, **kwargs)


class ConvGRUCell2d(ConvGRUCellNd):
  def __init__(self, in_size, out_size, hidden_size, kernel_size, **kwargs):
    super().__init__(in_size, out_size, hidden_size, kernel_size, N=2, **kwargs)


class ConvGRUCell3d(ConvGRUCellNd):
  def __init__(self, in_size, out_size, hidden_size, kernel_size, **kwargs):
    super().__init__(in_size, out_size, hidden_size, kernel_size, N=3, **kwargs)
