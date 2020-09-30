from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as func

from torch.nn.utils import spectral_norm

class InvertibleModule(nn.Module):
  r"""Invertible ``torch.nn.Module`` for normalizing flow generative models."""
  def __init__(self):
    super().__init__()
    self._inv = False

  def inv(self):
    r"""Returns the inverse of the given model."""
    result = copy(self)
    result._inv = True
    return result

  def det_J(self, *args, **kwargs):
    """Computes the log determinant of the ``InvertibleModule``'s Jacobian."""
    raise NotImplementedError("Abstract.")

  def action(self, *args, **kwargs):
    """Computes the forward action of the ``InvertibleModule``."""
    raise NotImplementedError("Abstract.")

  def inverse(self, *args, **kwargs):
    """Computes the inverse action of the ``InvertibleModule``."""
    raise NotImplementedError("Abstract.")

  def forward(self, *args, det_J=0.0, **kwargs):
    if self._inv:
      result = self.inverse(*args, **kwargs)
      inputs = result if isinstance(result, (list, tuple)) else [result]
      det = det_J - self.det_J(*result, **kwargs)
      return result, det
    else:
      return self.action(*args, **kwargs), det_J + self.det_J(*args, **kwargs)

class Split(InvertibleModule):
  r"""Splits an input tensor by its feature dimension.

  Args:
    keep (int): number of features in the first half of the split.

  Shapes:
    - Inputs: :math:`(B, F, ...)`
    - Keep: :math:`(B, Keep, ...)`
    - Drop: :math:`(B, F - Keep, ...)`
  """
  def __init__(self, keep):
    super().__init__()
    self.keep = keep

  def action(self, inputs):
    keep = inputs[:, :self.keep]
    drop = inputs[:, self.keep:]
    return keep, drop

  def inverse(self, keep, drop):
    return torch.cat((keep, drop), dim=1)

  def det_J(self, inputs):
    return 0.0

class Squeeze(InvertibleModule):
  r"""Groups features in tiles of a given kernel size.

  Args:
    kernel_size (int): size of the grouping kernel.

  Shapes:
    - Inputs: :math:`(B, F, W, H)`
    - Outputs: :math:`(B, F \cdot kernel, \frac{W}{kernel}, \frac{H}{kernel})`
  """
  def __init__(self, kernel_size=2):
    super().__init__()
    self.kernel_size = kernel_size

  def action(self, inputs):
    dim = inputs.dim()
    out = inputs
    for idx in range(dim - 2):
      out = out.reshape(
        *out.shape[:2 + idx],
        -1, self.kernel_size,
        *out.shape[3 + idx:]
      )
      out = out[:, :, None]
      out = out.transpose(2, 2 + idx + 2).squeeze(2 + idx + 2)
      out = out.reshape(out.shape[0], -1, *out.shape[3:])
    return out

  def inverse(self, inputs):
    dim = inputs.dim()
    out = inputs
    for idx in range(dim - 2):
      out = out.reshape(
        out.shape[0],
        -1, self.kernel_size,
        *out.shape[2:]
      )
      out = out.unsqueeze(-(idx + 1))
      out = out.transpose(2, -(idx + 1)).squeeze(2)
      rest = out.shape[-idx:] if idx > 0 else []
      out = out.reshape(
        *out.shape[:-(idx + 2)], -1,
        *rest
      )
    return out

  def det_J(self, _):
    return 0.0

class InvertibleAvgPool(InvertibleModule):
  r"""Performs invertible average pooling by splitting
  the input tensor into an averaged tensor and a tensor
  containing the difference from the average.

  Args:
    kernel_size (int): size of the average pooling kernel.

  Shape:
    - Inputs: :math:`(B, F, W, H)`
    - Mean: :math:`(B, F, \frac{W}{kernel}, \frac{H}{kernel})`
    - Difference: :math:`(B, F \cdot (kernel^2 - 1), \frac{W}{kernel}, \frac{H}{kernel})`
  """
  def __init__(self, kernel_size=2):
    super().__init__()
    self.kernel_size = kernel_size
    self.squeeze = Squeeze(kernel_size)

  def action(self, inputs):
    dim = inputs.dim() - 2
    pool = getattr(func, f"avg_pool{dim}d")
    squeezed = self.squeeze.action(inputs)
    mean = pool(inputs, self.kernel_size)
    alt_mean = squeezed.reshape(
      *mean.shape[:1], -1, *mean.shape[1:]
    ).sum(dim=1) / self.kernel_size ** dim
    mean_subtract = alt_mean.repeat_interleave(self.kernel_size ** dim - 1, dim=1)
    difference = squeezed[:, inputs.size(1):] - mean_subtract
    return alt_mean, difference

  def inverse(self, mean, difference):
    dim = mean.dim() - 2
    norm = self.kernel_size ** dim
    mean_add = mean.repeat_interleave(self.kernel_size ** dim - 1, dim=1)
    restored_difference = difference + mean_add
    difference_add = restored_difference.reshape(*mean.shape[:1], -1, *mean.shape[1:])
    difference_add = difference_add.sum(dim=1)
    restored_inputs = mean * norm - difference_add
    restored = torch.cat((restored_inputs, restored_difference), dim=1)
    out = self.squeeze.inverse(restored)
    return out

  def det_J(self, inputs):
    blocks = inputs.size(1)
    return blocks * self.kernel_size ** (inputs.dim() - 2)

class ActNorm(InvertibleModule):
  r"""Activation normalization reversible module for Glow.

  Args:
    size (int): number of input features.
  """
  def __init__(self, size):
    super(ActNorm, self).__init__()
    self.touched = False
    self.size = size
    self.scale = nn.Parameter(torch.zeros(size))
    self.bias = nn.Parameter(torch.zeros(size))

  def try_initialize_parameters(self, inputs):
    """Initializes module parameters setting mean = 0 and std = 1 for the
    first batch fed into the module."""
    if not self.touched:
      with torch.no_grad():
        reshaped = inputs.view(inputs.size(0), inputs.size(1), -1)
        reshaped = reshaped.permute(1, 0, 2).contiguous().view(self.size, -1)
        mean = reshaped.mean(dim=1)
        std = reshaped.std(dim=1)
        scale = 1 / std
        bias = - mean / std
        self.scale += scale
        self.bias += bias
    self.touched = True

  def det_J(self, inputs):
    reshaped = inputs.view(inputs.size(0), inputs.size(1), -1)
    size = reshaped.size(-1)
    return size * (torch.log(abs(self.scale))).sum()

  def action(self, inputs):
    if self.training:
      self.try_initialize_parameters(inputs)
    return inputs * self.scale + self.bias

  def inverse(self, inputs):
    return (inputs - self.bias) / self.scale

class AffineCoupling(InvertibleModule):
  r"""Affine coupling module for RealNVP and Glow. Applies an affine
  transformation parameterized by a subset of input features to the
  complement of that subset.

  Args:
    scale (nn.Module): module computing the scale component of the
      learned affine transformation.
    bias (nn.Module): module computing the bias component of the
      learned affine transformation.
    size (int): number of first size features to use to compute the
      learned affine transformation.
  """
  def __init__(self, scale, bias, size):
    super(AffineCoupling, self).__init__()
    self.size = size
    self.scale = scale
    self.bias = bias

  def det_J(self, inputs):
    context, inputs = inputs[:, :self.size], inputs[:, self.size:]
    scale = self.scale(context)
    return torch.log(abs(scale.exp())).sum()

  def action(self, inputs):
    context, inputs = inputs[:, :self.size], inputs[:, self.size:]
    scale = self.scale(context)
    bias = self.bias(context)
    result = inputs * scale.exp() + bias
    return torch.cat((context, result), dim=1)

  def inverse(self, inputs):
    context, inputs = inputs[:, :self.size], inputs[:, self.size:]
    scale = self.scale(context)
    bias = self.bias(context)
    result = (inputs - bias) / scale.exp()
    return torch.cat((context, result), dim=1)

class Exponential(InvertibleModule):
  r"""Implements an invertible transformation using an implicit
  matrix-free representation of the matrix exponential.

  Args:
    linear (nn.Module): module to apply as a matrix free implementation
      of a linear operator.
    steps (int): number of terms in the Taylor series expansion.
  """
  def __init__(self, linear, steps=10):
    super().__init__()
    self.linear = linear

  def action(self, inputs, *args, **kwargs):
    run = inputs
    out = inputs
    for idx in range(self.steps):
      run = self.linear(run, *args, **kwargs) / idx
      out = out + run
    return out

  def inverse(self, inputs, *args, **kwargs):
    run = inputs
    out = inputs
    for idx in range(self.steps):
      run = -self.linear(inputs, *args, **kwargs) / idx
      out = out + run
    return out

  def det_J(self, inputs, *args, **kwargs):
    raise NotImplementedError("Abstract.")

class LinearExponential(Exponential):
  r"""Matrix exponential of a plain linear operator with
  eigenvalues of magnitude at most 1.

  Args:
    in_size (int): number of input features.
    out_size (int): number of output features.
    steps (int): number of terms in the Taylor series expansion.
  """
  def __init__(self, in_size, out_size, steps=10):
    super().__init__(
      spectral_norm(nn.Linear(in_size, out_size, bias=False))
    )

  def det_J(self, inputs):
    return self.linear.weight.trace()

class ConvNdExponential(Exponential):
  r"""Matrix exponential of a general convolution operation with
  eigenvalues of magnitude at most 1.

  Args:
    in_size (int): number of input features.
    out_size (int): number of output features.
    kernel_size (int): size of the convolution kernel.
    steps (int): number of terms in the Taylor expansion.
  """
  def __init__(self, in_size, out_size, kernel_size=3, steps=10, dim=2):
    super().__init__(
      spectral_norm(getattr(nn, f"Conv{dim}d")(
        in_size, out_size, kernel_size,
        bias=False, padding=kernel_size // 2
      ))
    )
    self.dim = dim
    self.kernel_size = kernel_size

  def det_J(self, inputs):
    center = self.kernel_size // 2
    centered = self.linear.weight[(
      slice(None), slice(None), *(self.dim * [center])
    )]
    trace = centered.trace()
    return trace

class Conv1dExponential(ConvNdExponential):
  r"""Matrix exponential of a 1D convolution operation with
  eigenvalues of magnitude at most 1.

  Args:
    in_size (int): number of input features.
    out_size (int): number of output features.
    kernel_size (int): size of the convolution kernel.
    steps (int): number of terms in the Taylor expansion.
  """
  def __init__(self, in_size, out_size, kernel_size=3, steps=10):
    super().__init__(
      in_size, out_size, kernel_size=kernel_size, dim=1
    )

class Conv2dExponential(ConvNdExponential):
  r"""Matrix exponential of a 2D convolution operation with
  eigenvalues of magnitude at most 1.

  Args:
    in_size (int): number of input features.
    out_size (int): number of output features.
    kernel_size (int): size of the convolution kernel.
    steps (int): number of terms in the Taylor expansion.
  """
  def __init__(self, in_size, out_size, kernel_size=3, steps=10):
    super().__init__(
      in_size, out_size, kernel_size=kernel_size, dim=2
    )

class Conv3dExponential(ConvNdExponential):
  r"""Matrix exponential of a 3D convolution operation with
  eigenvalues of magnitude at most 1.

  Args:
    in_size (int): number of input features.
    out_size (int): number of output features.
    kernel_size (int): size of the convolution kernel.
    steps (int): number of terms in the Taylor expansion.
  """
  def __init__(self, in_size, out_size, kernel_size=3, steps=10):
    super().__init__(
      in_size, out_size, kernel_size=kernel_size, dim=3
    )
