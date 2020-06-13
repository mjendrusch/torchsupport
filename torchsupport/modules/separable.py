import torch
import torch.nn as nn
import torch.nn.functional as func

class DepthWiseSeparableConv1d(nn.Module):
  r"""Depthwise separable 1D convolution.

  Analogous functionality to :class:`torch.nn.Conv1d`.

  Args:
    in_channels (int): number of input channels.
    out_channels (int): number of output channels.
    kernel_size (int or (int, int)): kernel size.
    kwargs: additional keyword arguments. See `Conv1d` for details. 
  """
  def __init__(self, in_channels, out_channels, kernel_size,
               stride=1, padding=0, dilation=1, bias=True):
    super(DepthWiseSeparableConv1d, self).__init__()
    self.depth_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                                stride=stride, padding=padding,
                                dilation=dilation, bias=bias)
    self.point_conv = nn.Conv1d(in_channels, out_channels, 1)

  def forward(self, inputs):
    return self.point_conv(self.depth_conv(inputs))

class DepthWiseSeparableConv2d(nn.Module):
  r"""Depthwise separable 2D convolution.

  Analogous functionality to :class:`torch.nn.Conv2d`.

    Args:
      in_channels (int): number of input channels.
      out_channels (int): number of output channels.
      kernel_size (int or (int, int)): kernel size.
      kwargs: additional keyword arguments. See `Conv2d` for details. 
  """
  def __init__(self, in_channels, out_channels, kernel_size,
               stride=1, padding=1, dilation=1, bias=True):
    super(DepthWiseSeparableConv2d, self).__init__()
    self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                                stride=stride, padding=padding,
                                dilation=dilation, bias=bias)
    self.point_conv = nn.Conv2d(in_channels, out_channels, 1)

  def forward(self, inputs):
    return self.point_conv(self.depth_conv(inputs))
