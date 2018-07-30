import torch.nn as nn
import torch.nn.functional as func

class Conv1x1(nn.Module):
  """Submodule encoding convolution plus 1x1 convolution."""
  def __init__(self, width, stride, input, kernels, kernels11,
               activation=func.leaky_relu,
               activation_1x1=func.leaky_relu,
               dim=2):
    super(Conv1x1, self).__init__()
    assert(dim in [1, 2, 3])
    self.conv_op = nn.__dict__[f"Conv{dim}d"]
    self.bn_op = nn.__dict__[f"BatchNorm{dim}d"]
    self.conv = self.conv_op(input, kernels, width, stride, 1)
    self.bn = self.bn_op(kernels)
    self.x11 = self.conv_op(kernels, kernels11, 1, 1)
    self.bn11 = self.bn_op(kernels11)
    self.activation = activation
    self.activation_1x1 = activation_1x1

  def forward(self, x):
    x = self.bn(self.activation(self.conv(x)))
    x = self.bn11(self.activation_1x1(self.x11(x)))
    return x

class MobileConv(nn.Module):
  """"""
  def __init__(self, width, stride, input, kernels, kernels11):
    pass

  def forward(self, x):
    pass