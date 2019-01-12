import torch
import torch.nn as nn
import torch.nn.functional as func

class DepthWiseSeparableConv1d(nn.Module):
  def __init__(in_channels, out_channels, kernel_size,
               stride=1, padding=0, dilation=1, bias=True):
    """Depthwise separable 1D convolution.

    Args:
      in_channels (int): number of input channels.
      out_channels (int): number of output channels.
      kernel_size (int or (int, int)): kernel size.
      kwargs: additional keyword arguments. See `Conv1d` for details. 
    """
    self.depth_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                                stride=stride, padding=padding,
                                dilation=dilation, bias=bias)
    self.point_conv = nn.Conv1d(in_channels, out_channels, 1)

  def forward(self, input):
    return self.point_conv(self.depth_conv(input))

class DepthWiseSeparableConv2d(nn.Module):
  def __init__(in_channels, out_channels, kernel_size,
               stride=1, padding=0, dilation=1, bias=True):
    """Depthwise separable 2D convolution.

    Args:
      in_channels (int): number of input channels.
      out_channels (int): number of output channels.
      kernel_size (int or (int, int)): kernel size.
      kwargs: additional keyword arguments. See `Conv2d` for details. 
    """
    self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                                stride=stride, padding=padding,
                                dilation=dilation, bias=bias)
    self.point_conv = nn.Conv2d(in_channels, out_channels, 1)

  def forward(self, input):
    return self.point_conv(self.depth_conv(input))
