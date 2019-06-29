import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.structured.modules import basic as snn

class AdjacencyAction(snn.ConnectedModule):
  def reduce(self, data, message):
    return (data + message.sum(dim=0)) / (message.size(0) + 1)

class LaplacianAction(snn.ConnectedModule):
  def __init__(self, normalized=False):
    super(LaplacianAction, self).__init__()
    self.normalized = normalized

  def reduce(self, data, message):
    factor = 1
    if self.normalized:
      factor = 1 / message.size(0)
    return factor * (message.size(0) * data - message.sum(dim=0))

class GCN(nn.Module):
  def __init__(self, in_size, out_size, depth=1, activation=func.relu):
    super(GCN, self).__init__()
    self.linear = nn.Linear(in_size, out_size)
    self.connected = AdjacencyAction()
    self.activation = activation
    self.depth = depth

  def forward(self, data, structure):
    out = self.linear(data)
    for _ in range(self.depth):
      out = self.connected(out, out, structure)
    return self.activation(out)

class Chebyshev(nn.Module):
  def __init__(self, in_size, out_size, depth=1, activation=func.relu):
    super(Chebyshev, self).__init__()
    self.linear = nn.Linear(in_size, out_size)
    self.connected = LaplacianAction(normalized=True)
    self.activation = activation
    self.depth = depth

  def forward(self, data, structure):
    out_2 = self.linear(data)
    out_1 = self.connected(out_2, out_2, structure)
    out = out_1 + out_2
    for _ in range(self.depth):
      tmp = out_1
      out_1 = 2 * self.connected(out_1, out_1, structure) - out_2
      out_2 = tmp
      out += out_1
    return self.activation(out)

class ConvSkip(nn.Module):
  def __init__(self, in_size, out_size,
               merge_size, activation=func.relu,
               connected=LaplacianAction(normalized=True)):
    super(ConvSkip, self).__init__()
    self.transform = nn.Linear(merge_size, out_size)
    self.linear = nn.Linear(in_size, out_size)
    self.activation = activation
    self.connected = connected

  def forward(self, data, merge, structure):
    out = self.linear(data)
    out = self.connected(out, out, structure)
    return self.activation(out + self.transform(merge))

class WideConvSkip(nn.Module):
  def __init__(self, in_size, out_size, merge_size,
               width=3, activation=func.relu,
               connected=LaplacianAction(normalized=True)):
    super(WideConvSkip, self).__init__()
    self.transform = nn.Linear(merge_size, out_size * width)
    self.linear = nn.Conv1d(in_size * width, out_size * width, 1, groups=width)

  def forward(self, data, merge, structure):
    out = self.linear(data.unsqueeze(1)).squeeze()
    out = self.connected(out, out, structure)
    return self.activation(out + self.transform(merge))

class ARMA(nn.Module):
  def __init__(self, in_size, out_size, hidden_size,
               width=3, depth=3, share=False, activation=func.relu,
               connected=LaplacianAction(normalized=True)):
    super(ARMA, self).__init__()
    self.width = width
    self.preprocess = ConvSkip(
      in_size, hidden_size * width, in_size,
      connected=connected, activation=activation
    )

    if share:
      shared_block = WideConvSkip(
        hidden_size, hidden_size, in_size,
        width=width, connected=connected,
        activation=activation
      )
      self.blocks = nn.ModuleList([shared_block for _ in range(depth - 2)])
    else:
      self.blocks = nn.ModuleList([
        WideConvSkip(
          hidden_size, hidden_size, in_size,
          width=width, connected=connected,
          activation=activation
        )
        for _ in range(depth - 2)
      ])
    self.postprocess = WideConvSkip(
      hidden_size, out_size, in_size,
      width=width, connected=connected,
      activation=activation
    )

  def forward(self, data, structure):
    out = self.preprocess(data, data, structure)
    for block in self.blocks:
      out = block(out, data, structure)
    out = self.postprocess(out, data, structure)
    out = out.reshape(data.size(0), -1, self.width)
    return func.adaptive_avg_pool1d(out, 1)

class APP(nn.Module):
  def __init__(self, in_size, out_size,
               depth=10, teleport=0.5, activation=func.relu,
               connected=LaplacianAction(normalized=True)):
    super(APP, self).__init__()
    self.linear = nn.Linear(in_size, out_size)
    self.teleport = teleport
    self.depth = depth
    self.activation = activation
    self.connected = connected

  def forward(self, data, structure):
    embedding = self.linear(data)
    out = embedding
    for _ in range(self.depth):
      out = (1 - self.teleport) * self.connected(out, out, structure)
      out += self.teleport * embedding
    return self.activation(out)

class MultiScaleAPP(nn.Module):
  def __init__(self, in_size, out_size,
               depth=10, teleports=[0.1, 0.2, 0.3],
               activation=func.relu,
               connected=LaplacianAction(normalized=True)):
    super(MultiScaleAPP, self).__init__()
    self.source_attention = nn.Linear(in_size, out_size)
    self.target_attention = nn.Linear(out_size, out_size)
    self.scales = nn.ModuleList([
      APP(
        in_size, out_size,
        depth=depth, teleport=teleport,
        activation=activation,
        connected=connected
      )
      for teleport in teleports
    ])

  def forward(self, data, structure):
    scales = [
      scale(data, structure)
      for scale in self.scales
    ]
    source_attention = self.source_attention(data)
    scale_attention = torch.softmax(torch.cat([
      self.target_attention(scale).dot(source_attention)
      for scale in self.scales
    ], dim=1), dim=1)
    scales = torch.cat([
      scale.unsqueeze(1)
      for scale in scales
    ], dim=1)
    return (scale_attention * scales).sum(dim=1)
