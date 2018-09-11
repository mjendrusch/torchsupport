import torch
import torch.nn as nn
import torch.nn.functional as func

class DynamicOp(nn.Module):
  def __init__(self, op, generator):
    self.generator = generator
    self.op = op

  def forward(self, x, generation_input):
    weights = self.generator(generation_input)
    size = x.size()
    batch_size = size[0]
    result = torch.Tensor(size[0], size[1], size[2], size[3])
    for idx in range(x.shape[0]):
      result[idx, :, :, :] = self.op(x[idx, :, :, :])
    return result

class DynamicConv2d(nn.Module):
  def __init__(self, generator, kernel_size, stride=1,
               padding=0, dilation=1):
    """Performs an efficient dynamic convolution using grouped convs.
    
    Arguments
    ---------

    generator : an `torch.nn.Module` weight generating network, returning weights
                of the shape `(batch_size, output_channels, input_channels, width, height)`.
    """
    self.generator = generator
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation

  def forward(self, input, generation_input):
    size = input.size()
    batch_size = size[0]
    channels = size[1]

    weights = self.generator(generation_input)
    weight_size = weights.size()
    weights.reshape(batch_size * weight_size[0], weight_size[1], weight_size[2], weight_size[3])
    weight_size = weights.size()

    # weight size resolution:
    assert(weight_size[0] % batch_size == 0, "Filter size not an integer multiple of the batch size!")
    out_channels = weight_size[0] // batch_size
    filters = weight_size[1]
    width = weight_size[2]
    height = weight_size[3]

    filters = weight_size[1]
    debatched = input.reshape(channels * batch_size, size[2], size[3])
    result = nn.conv2d(debatched, weights,
                       stride=self.stride,
                       padding=self.padding,
                       dilation=self.dilation,
                       groups=batch_size)
    result = result.reshape(batch_size, filters, result.size()[2], result.size()[3])
    return result

class ParametricOp(nn.Module):
  def __init__(self, op):
    self.op = op

  def forward(self, input, generator_input):
    return self.op(input, generator_input)

class Merge(nn.Module):
  def __init__(self, operation=lambda x, shape: x):
    self.operation = operation

  def forward(self, x, y):
    new_y = self.operation(y, x.shape)
    return torch.cat(x, new_y)