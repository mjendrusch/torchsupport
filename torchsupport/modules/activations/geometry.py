import torch

def torus(sin, cos):
  r"""Pins a pair of N-dimensional coordinates to
  proper coordinates on the N-torus.

  Args:
    sin (torch.Tensor): unnormalized sine.
    cos (torch.Tensor): unnormalized cosine.

  Returns:
    Corresponding coordinates on a torus.
  """
  return torch.atan2(sin, cos)
