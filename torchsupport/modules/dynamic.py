import torch
import torch.nn as nn
import torch.nn.functional as func

def _dynamic_convnd(inputs, weight, bias=None, N=2, **kwargs):
  conv = getattr(func, f"conv{N}d")
  batch_size = weight.size(0)
  inputs = inputs.view(-1, *inputs.shape[2:])
  weight = weight.view(-1, *weight.shape[2:])
  if bias:
    bias = bias.view(-1)
  result = conv(inputs, weight, bias=bias, groups=batch_size, **kwargs)
  result = result.view(batch_size, -1, *result.shape[1:])
  return result

def dynamic_conv1d(inputs, weight, bias=None, **kwargs):
  r"""Dynamic 1d convolution. For details, see `torch.nn.functional.conv1d`

  Args:
    inputs (torch.Tensor :math:`(B, C_i, W)`): input tensor.
    weight (torch.Tensor :math:`(B, C_o, C_i, K)`): batch of weight tensors.
    bias (torch.Tensor :math:`B, C_o`): batch of bias tensors.
  """
  return _dynamic_convnd(inputs, weight, bias=bias, N=1, **kwargs)

def dynamic_conv2d(inputs, weight, bias=None, **kwargs):
  r"""Dynamic 2d convolution. For details, see `torch.nn.functional.conv2d`

  Args:
    inputs (torch.Tensor :math:`(B, C_i, H, W)`): input tensor.
    weight (torch.Tensor :math:`(B, C_o, C_i, K_H, K_W)`): batch of weight tensors.
    bias (torch.Tensor :math:`B, C_o`): batch of bias tensors.
  """
  return _dynamic_convnd(inputs, weight, bias=bias, N=2, **kwargs)

def dynamic_conv3d(inputs, weight, bias=None, **kwargs):
  r"""Dynamic 3d convolution. For details, see `torch.nn.functional.conv3d`

  Args:
    inputs (torch.Tensor :math:`(B, C_i, X, Y, Z)`): input tensor.
    weight (torch.Tensor :math:`(B, C_o, C_i, K_X, K_Y, K_Z)`): batch of weight tensors.
    bias (torch.Tensor :math:`B, C_o`): batch of bias tensors.
  """
  return _dynamic_convnd(inputs, weight, bias=bias, N=3, **kwargs)

def dynamic_linear(inputs, weight, bias=None):
  r"""Dynamic linear layer. For details, see `torch.nn.functional.linear`

  Args:
    inputs (torch.Tensor :math:`(B, C_i, W)`): input tensor.
    weight (torch.Tensor :math:`(B, C_o, C_i)`): batch of weight tensors.
    bias (torch.Tensor :math:`B, C_o`): batch of bias tensors.
  """
  result = torch.bmm(inputs[:, None], weight.transpose(1, 2))[:, 0]
  if bias:
    result = result + bias
  return result
