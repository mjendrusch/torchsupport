import torch
import torch.nn as nn
import torch.nn.functional as func
from torchsupport.networks.unet import *
from torchsupport.modules.multiscale import Autoscale

class UNetModel(nn.Module):
  def __init__(self, depth, dilated, attention, max_classes, in_channels, is_inverted=False):
    super(UNetModel, self).__init__(self)
    self.depth = depth
    self.dilated = dilated
    self.attention = attention
    self.max_classes = max_classes
    self.in_channels = in_channels
    self.is_inverted = is_inverted
    ublock = StandardUNetConv
    if dilated:
      ublock = DilatedUNetConv
    first_filters = 6

    self.unet = UNet(down_block=ublock, up_block=ublock, in_channels=in_channels,
                     depth=depth, first_filters=first_filters)
    self.postprocessor = nn.Conv2d(2**first_filters, max_classes, 1)

  def wnet_decoder(self):
    result = UNetModel(self.depth, self.dilated, self.attention,
                       self.in_channels, is_inverted=True)
    return result

  def forward(self, input):
    out = self.unet(input)
    out = self.postprocessor(out)
    if not self.is_inverted:
      out = func.softmax(out, dim=1)
    return out

class AutofocusResBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(AutofocusResBlock, self).__init__(self)
    self.dm_0 = DilatedMultigrid(in_channels, out_channels, 3, levels=[0,1,2,4])
    self.auto_0 = Autoscale(self.dm_0)
    self.dm_1 = DilatedMultigrid(out_channels, out_channels, 3, levels=[0,1,2,4], activation=lambda x: x)
    self.auto_1 = Autoscale(self.dm_1)

  def forward(self, input):
    out = self.auto_0(input)
    out = self.auto_1(output)
    out = out + input
    out = func.relu(out)
    return out

class AutofocusModel(nn.Module):
  def __init__(self, depth, max_classes, in_channels, is_inverted=False):
    super(AutofocusModel, self).__init__(self)
    self.depth = depth
    self.max_classes = max_classes
    self.in_channels = in_channels
    self.is_inverted = is_inverted

    self.conv_0 = nn.Sequential(
      nn.Conv2d(in_channels, 30, 3),
      nn.BatchNorm2d(),
      nn.ReLU()
    )
    self.conv_1 = nn.Sequential(
      nn.Conv2d(30, 30, 3),
      nn.BatchNorm2d(),
      nn.ReLU()
    )
    self.resblocks = []
    for idx in range(depth):
      self.resblocks.append(AutofocusResBlock(
        30 + idx * 10, 30 + (idx + 1) * 10
      ))
    self.postprocessor = nn.Conv2d(30 + (depth + 1) * 10, max_classes, 1)

  def wnet_decoder(self, upsample=False):
    result = AutofocusModel(self.depth, self.in_channels,
                            self.max_classes, is_inverted=True)
    if upsample:
      result = nn.Sequential(
        nn.UpsamplingBilinear(size=(224, 224)),
        result
      )

  def forward(self, input):
    out = self.conv_0(input)
    out = self.conv_1(out)
    for block in self.resblocks:
      out = block(out)
    out = self.postprocessor(out)
    if not self.is_inverted:
      out = func.softmax(out, dim=1)
    return out

class ScalePreprocessor(nn.Module):
  def __init__(self):
    super(ScalePreprocessor, self).__init__()
    self.conv_0 = nn.Conv2d(3, 16, 3)
    self.bn_0 = nn.BatchNorm2d(16)
    self.conv_1 = nn.Conv2d(16, 3, 1)
    self.bn_1 = nn.BatchNorm2d(3)

  def forward(self, input):
    out = self.conv_0(input)
    out = self.bn_0(out)
    out = func.relu_(out)
    out = self.conv_1(out)
    out = self.bn_1(out)
    out = func.relu(out + input)
    return out

class MultiScaleModel(nn.Module):
  def __init__(self, inner_model, scales):
    super(MultiScaleModel, self).__init__()
    self.inner_model = inner_model
    self.scales = scales
    self.scale_preprocessors = nn.ModuleList([
      ScalePreprocessor()
      for _ in range(scales)
    ])

  def wnet_decoder(self):
    pass # TODO

  def forward(self, input):
    scales = [
      self.scale_preprocessors[idx](
        input[:, idx * 3:(idx + 1) * 3, :, :]
      )
      for idx in range(self.scales)
    ]
    out = torch.cat(scales, dim=1)
    return self.inner_model(out)
