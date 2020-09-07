import torch
import torch.nn as nn
import torch.nn.functional as func

class MaskedConv1d(nn.Conv1d):
  r"""1D causally masked convolution for autoregressive
  convolutional models.

  Thin wrapper around :class:`nn.Conv1d`."""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    mask = torch.zeros_like(self.weight)
    center_x = self.kernel_size[0] // 2
    for idx in range(self.kernel_size[0]):
      mask[:, :, idx] = int(idx < center_x)
    self.mask = mask

  def forward(self, inputs):
    self.weight.data = self.mask * self.weight.data
    return super().forward(inputs)

class MaskedConv2d(nn.Conv2d):
  r"""2D causally masked convolution for autoregressive
  convolutional models.

  Thin wrapper around :class:`nn.Conv2d`."""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    mask = torch.zeros_like(self.weight)
    center_x = self.kernel_size[0] // 2
    center_y = self.kernel_size[1] // 2
    for idx in range(self.kernel_size[0]):
      for idy in range(self.kernel_size[1]):
        mask[:, :, idx, idy] = int(idx < center_x or (idx == center_x and idy < center_y))
    self.mask = mask

  def forward(self, inputs):
    self.weight.data = self.mask * self.weight.data
    return super().forward(inputs)

class MaskedConv3d(nn.Conv3d):
  r"""3D causally masked convolution for autoregressive
  convolutional models.

  Thin wrapper around :class:`nn.Conv3d`."""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    mask = torch.zeros_like(self.weight)
    center_x = self.kernel_size[0] // 2
    center_y = self.kernel_size[1] // 2
    center_z = self.kernel_size[2] // 2
    for idx in range(self.kernel_size[0]):
      for idy in range(self.kernel_size[1]):
        for idz in range(self.kernel_size[2]):
          mask[:, :, idx, idy, idz] = int(
            idx < center_x or \
              (idx == center_x and idy < center_y) or \
                (idx == center_x and idy == center_y and idz < center_z)
          )
    self.mask = mask

  def forward(self, inputs):
    self.weight.data = self.mask * self.weight.data
    return super().forward(inputs)
