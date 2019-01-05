import torch
import torch.nn as nn
import torch.nn.functional as func
from torchsupport.modules.separable import DepthWiseSeparableConv2d

class WNet(nn.Module):
  def __init__(self, down_block=None, up_block=None, n_classes=2,
               in_channels=1, depth=5, first_filters=6, up_mode='upconv'):
    """
    Template for general WNet-like neural networks for unsupervised semantic segmentation.
    Args:
      down_block (callable): downsampling block template
      up_block (callable): upsampling block template
      n_classes (int): number of segmentation classes
      in_channels (int): number of input channels
      depth (int): depth of the network
      first_filters (int): number of filters in the first layer is 2**first_filters
      up_mode (str): one of 'upconv' or 'upsample'.
                      'upconv' will use transposed convolutions for
                      learned upsampling.
                      'upsample' will use bilinear upsampling.
    """
    super(WNet, self).__init__()
    assert up_mode in ('upconv', 'upsample')
    self.down_block = down_block
    self.up_block = up_block
    if self.down_block == None:
      self.down_block = StandardWNetDown
    if self.up_block == None:
      self.up_block = StandardWNetUp
    seg_channels = 2 ** first_filters
    unseg_channels = 2 ** first_filters
    self.seg_unet = UNet(down_block, up_block, n_classes,
                         in_channels=in_channels, depth=depth,
                         first_filters=first_filters, up_mode=up_mode)
    self.unseg_unet = UNet(down_block, up_block, n_classes,
                           in_channels=in_channels, depth=depth,
                           first_filters=first_filters, up_mode=up_mode)
    self.seg_layer = nn.Conv2d(seg_channels, n_classes, 1)
    self.rec_layer = nn.Conv2d(unseg_channels, in_channels, 1)

  def infer(self, input):
    out = self.seg_unet(input)
    out = self.seg_layer(out)
    out = func.softmax(out)
    return out

  def reconstruct(self, input):
    out = self.unseg_unet(input)
    out = self.rec_layer(out)
    return out
  
  def forward(self, input):
    return self.reconstruct(self.infer(input))

class StandardWNetDown(nn.Module):
  def __init__(self, in_channels, out_channels, position):
    """
    Default down convolution block for the WNet.
    Args:
      in_channels (int): number of input channels.
      out_channels (int): number of output channels.
      position (int): position of the block within the WNet.
    """
    super(StandardWNetDown, self).__init__()
    if position == 0:
      self.block_0 = nn.Conv2d(in_channels, out_channels, 3)
      self.block_1 = nn.Conv2d(in_channels, out_channels, 3)
    else:
      self.block_0 = DepthWiseSeparableConv2d(in_channels, out_channels, 3)
      self.block_1 = DepthWiseSeparableConv2d(out_channels, out_channels, 3)

  def forward(self, input):
    return self.block_1(self.block_0(input))

class StandardWNetUp(nn.Module):
  def __init__(self, in_channels, out_channels, position):
    """
    Default up convolution block for the WNet.
    Args:
      in_channels (int): number of input channels.
      out_channels (int): number of output channels.
      position (int): position of the block within the WNet.
    """
    super(StandardWNetUp, self).__init__()
    if position == 0:
      self.block_0 = nn.Conv2d(in_channels, out_channels, 3)
      self.block_1 = nn.Conv2d(in_channels, out_channels, 3)
    else:
      self.block_0 = DepthWiseSeparableConv2d(in_channels, out_channels, 3)
      self.block_1 = DepthWiseSeparableConv2d(out_channels, out_channels, 3)

  def forward(self, input):
    return self.block_1(self.block_0(input))

class UNet(nn.Module):
  def __init__(self, down_block=StandardUNetConv, up_block=StandardUNetConv, pooling=func.avg_pool2d, in_channels=1, depth=5, first_filters=6, up_mode='upconv'):
    """
    Template for general U-Net-like architectures.
    Args:
        down_block (callable): downsampling block template
        up_block (callable): upsampling block template
        in_channels (int): number of input channels
        depth (int): depth of the network
        first_filters (int): number of filters in the first layer is 2**first_filters
        up_mode (str): one of 'upconv' or 'upsample'.
                        'upconv' will use transposed convolutions for
                        learned upsampling.
                        'upsample' will use bilinear upsampling.
    """
    super(UNet, self).__init__()
    assert up_mode in ('upconv', 'upsample')
    self.depth = depth
    self.pooling = pooling
    self.down_path = UNet.down_part(down_block, pooling, in_channels, depth, first_filters)
    self.up_path = UNet.up_part(up_block, pooling, in_channels, depth, first_filters, up_mode)

  @staticmethod
  def down_part(down_block=StandardUNetConv, pooling=func.avg_pool2d, in_channels=1, depth=5, first_filters=6):
    prev_channels = in_channels
    result = nn.ModuleList()
    for i in range(depth):
      result.append(UNetDownBlock(down_block, prev_channels, 2**(first_filters+i), i))
      prev_channels = 2**(first_filters+i)
    return UNetDownPart(result)
  
  @staticmethod
  def up_part(up_block=StandardUNetConv, pooling=func.avg_pool2d, in_channels=1, depth=5, first_filters=6, up_mode='upconv'):
    prev_channels = 2**(first_filters+depth-1)
    result = nn.ModuleList()
    for i in reversed(range(depth - 1)):
      result.append(UNetUpBlock(up_block, prev_channels, 2**(first_filters+i), up_mode, i))
      prev_channels = 2**(first_filters+i)
    return UNetUpPart(result)

  def forward(self, x):
    x, blocks = self.down_part(x)
    x = self.up_part(x, blocks)
    return x

class UNetDownPart(nn.Module):
  def __init__(self, module_list):
    self.modules = module_list

  def forward(self, x):
    blocks = []
    for i, down in enumerate(self.modules):
      x = down(x)
      if i != len(self.modules)-1:
        blocks.append(x)
        x = self.pooling(x, 2)
    return x, blocks

class UNetUpPart(nn.Module):
  def __init__(self, module_list):
    self.modules = module_list

  def forward(self, x, blocks):
    for i, up in enumerate(self.modules):
      x = up(x, blocks[-i-1])
    return x

class UNetDownBlock(nn.Module):
  def __init__(self, down_block, in_size, out_size, position):
    """
    (INTERNAL) Implementation of a UNet down block.
    """
    super(UNetDownBlock, self).__init__()
    self.block = down_block(in_size, out_size, position)

  def forward(self, x):
    return self.block(x)

class UNetUpBlock(nn.Module):
  def __init__(self, up_block, in_size, out_size, up_mode, position):
    """
    (INTERNAL) Implementation of a UNet up block.
    """
    super(UNetUpBlock, self).__init__()
    if up_mode == 'upconv':
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
    elif up_mode == 'upsample':
        self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                up_block)
    self.block = up_block(in_size, out_size, position)

  def center_crop(self, layer, target_size):
    _, _, layer_height, layer_width = layer.size()
    diff_y = up_block, (layer_height - target_size[0]) // , up_mode[1]) // 2
    return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

  def forward(self, x, bridge):
    up = self.up(x)
    crop1 = self.center_crop(bridge, up.shape[2:])
    out = torch.cat([up, crop1], 1)
    return self.block(out)

class StandardUNetConv(nn.Module):
  def __init__(self, in_channels, out_channels, position):
    """
    Default UNet convolution.
    Args:
      in_channels (int): number of input channels.
      out_channels (int): number of output channels.
      position (int): position within the UNet.
    """
    self.block_0 = nn.Conv2d(in_channels, out_channels, 3)
    self.block_1 = nn.Conv2d(out_channels, out_channels, 3)

  def forward(self, input):
    return self.block_1(self.block_0(input))
