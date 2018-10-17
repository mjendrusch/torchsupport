import torch.nn as nn
import torch.nn.functional as func
import torchsupport.modules.reduction as red

class Conv1x1(nn.Module):
  r"""Submodule encoding convolution plus 1x1 convolution.
  
  Arguments:
    width (int) : integer kernel width.
    stride (int) : integer convolution stride.
    input (int) : number of input features.
    kernels (int) : number of standard kernels.
    kernels11 (int) : number of 1-by-1 convolution kernels.
    activation (function) : activation of the standard convolution layers, defaults to `leaky_relu`.
    activation_1x1 (function) : activation of the 1-by-1 convolution kernels.
  """
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

class UpConv1x1(nn.Module):
  r"""Submodule encoding convolution plus 1x1 convolution.
  
  Arguments:
    width (int) : integer kernel width.
    stride (int) : integer convolution stride.
    input (int) : number of input features.
    kernels (int) : number of standard kernels.
    kernels11 (int) : number of 1-by-1 convolution kernels.
    activation (function) : activation of the standard convolution layers, defaults to `leaky_relu`.
    activation_1x1 (function) : activation of the 1-by-1 convolution kernels.
  """
  def __init__(self, width, stride, input, kernels, kernels11,
               activation=func.leaky_relu,
               activation_1x1=func.leaky_relu,
               dim=2):
    super(UpConv1x1, self).__init__()
    assert(dim in [1, 2, 3])
    self.upsampling = nn.__dict__[f"UpsamplingBilinear{dim}d"]
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

def DownUp1x1(width, stride, input, kernels, kernels11,
              activation=func.leaky_relu,
              activation_1x1=func.leaky_relu,
              combination=lambda x, y: torch.cat([x, y], dim=1),
              dim=2):
  down = Conv1x1(width, stride, input, kernels, kernels11,
                 activation=activation,
                 activation_1x1=activation_1x1,
                 dim=dim)
  up = UpConv1x1(width, stride, kernels11 + input, kernels, kernels11,
                 activation=activation,
                 activation_1x1=activation_1x1,
                 dim=dim)
  sdown, sup = res.shortcut(down, up, combination=combination)
  return sdown, sup
  
class MobileConv(nn.Module):
  """"""
  def __init__(self, width, stride, input, kernels, kernels11):
    pass

  def forward(self, x):
    pass