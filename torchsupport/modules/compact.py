import torch.nn as nn
import torch.nn.functional as func

class Conv1x1(nn.Module):
  """Submodule encoding convolution plus 1x1 convolution."""
  def __init__(self, width, stride, input, kernels, kernels11,
               activation=func.leaky_relu,
               activation_1x1=func.leaky_relu):
    super(Conv1x1, self).__init__()
    self.conv = nn.Conv2d(input, kernels, width, stride, 1)
    self.bn = nn.BatchNorm2d(kernels)
    self.x11 = nn.Conv2d(kernels, kernels11, 1, 1)
    self.bn11 = nn.BatchNorm2d(kernels11)
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