import torch
import torch.nn as nn
import torch.nn.functional as func

class Poly(nn.Module):
  r"""Learnable polynomial transformation.

  Uses arbitrary :class:`nn.Module` components to compute a learnable
  polynomial of arbitrary rank in its inputs.

  Args:
    in_size (int): number of input features.
    out_size (int): number of output features.
    hidden_size (int): number of hidden features. Default: 128
    depth (int): rank of the computed polynomial. Default: 2
    inputs_kwargs (dict): keyword arguments forwarded to the
      input-transforming module.
    internal_kwargs (dict): keyword arguments forwarded to the
      internal representation-transforming module.
  """
  def __init__(self, in_size, out_size, hidden_size=128, depth=2,
               input_kwargs=None, internal_kwargs=None):
    super().__init__()
    self.depth = depth
    self.input_blocks = nn.ModuleList([
      self.make_block(in_size, hidden_size, **input_kwargs)
      for idx in range(depth)
    ])
    self.internal_blocks = nn.ModuleList([
      nn.Identity()
    ] + [
      self.make_block(hidden_size, hidden_size, **internal_kwargs)
      for idx in range(depth - 1)
    ])
    self.internal_constants = nn.ParameterList([
      self.make_constant(hidden_size)
      for idx in range(depth)
    ])

    self.output_block = self.make_block(hidden_size, out_size, **internal_kwargs)
    self.output_constant = self.make_constant(out_size)

  def make_block(self, in_size, out_size, **kwargs):
    r"""Creates a neural network block transforming inputs or internal
    representations.

    Args:
      in_size (int): number of input features.
      out_size (int): number ot output features.
      kwargs (dict): keyword arguments used in network construction.
    """
    raise NotImplementedError("Abstract")

  def make_constant(self, size):
    r"""Creates a learnable constant in the polynomial.

    Args:
      size (int): number of features for the constant.
    """
    raise NotImplementedError("Abstract")

  def forward(self, inputs):
    out = 0.0
    for (block, int_block, int_const) in zip(
        self.input_blocks, self.internal_blocks, self.internal_constants
    ):
      out = (int_block(out) + int_const + 1) * block(inputs) + out
    out = self.output_block(out) + self.output_constant
    return out

class PolyConv1d(Poly):
  r"""1D Convolutional polynomial neural network.

  Args:
    in_size (int): number of input features.
    out_size (int): number of output features.
    hidden_size (int): number of hidden features. Default: 128
    depth (int): rank of the computed polynomial. Default: 2
    inputs_kwargs (dict): keyword arguments forwarded to the
      input-transforming module
      (for possible options see: :class:`nn.Conv1d`).
    internal_kwargs (dict): keyword arguments forwarded to the
      internal representation-transforming module
      (for possible options see: :class:`nn.Conv1d`).
  """
  def make_block(self, in_size, out_size, **kwargs):
    return nn.Sequential(
      nn.Conv1d(in_size, out_size, 3, **kwargs),
      nn.ReLU()
    )

  def make_constant(self, size):
    return nn.Parameter(torch.randn(1, size, 1, requires_grad=True) / size)

class PolyConv2d(Poly):
  r"""2D Convolutional polynomial neural network.

  Args:
    in_size (int): number of input features.
    out_size (int): number of output features.
    hidden_size (int): number of hidden features. Default: 128
    depth (int): rank of the computed polynomial. Default: 2
    inputs_kwargs (dict): keyword arguments forwarded to the
      input-transforming module
      (for possible options see: :class:`nn.Conv2d`).
    internal_kwargs (dict): keyword arguments forwarded to the
      internal representation-transforming module
      (for possible options see: :class:`nn.Conv2d`).
  """
  def make_block(self, in_size, out_size, **kwargs):
    return nn.Sequential(
      nn.Conv2d(in_size, out_size, 3, **kwargs),
      nn.ReLU()
    )

  def make_constant(self, size):
    return nn.Parameter(torch.randn(1, size, 1, 1, requires_grad=True) / size)

class PolyConv3d(Poly):
  r"""3D Convolutional polynomial neural network.

  Args:
    in_size (int): number of input features.
    out_size (int): number of output features.
    hidden_size (int): number of hidden features. Default: 128
    depth (int): rank of the computed polynomial. Default: 2
    inputs_kwargs (dict): keyword arguments forwarded to the
      input-transforming module
      (for possible options see: :class:`nn.Conv3d`).
    internal_kwargs (dict): keyword arguments forwarded to the
      internal representation-transforming module
      (for possible options see: :class:`nn.Conv3d`).
  """
  def make_block(self, in_size, out_size, **kwargs):
    return nn.Sequential(
      nn.Conv3d(in_size, out_size, 3, **kwargs),
      nn.ReLU()
    )

  def make_constant(self, size):
    return nn.Parameter(torch.randn(1, size, 1, 1, 1, requires_grad=True) / size)
