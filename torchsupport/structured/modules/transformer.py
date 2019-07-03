import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.basic import MLP
from torchsupport.structured.modules.basic import NeighbourDotMultiHeadAttention

class Transformer(nn.Module):
  def __init__(self, in_size, out_size, hidden_size,
               query_size=None, attention_size=64,
               heads=8, mlp_depth=3, activation=func.elu_):
    super(Transformer, self).__init__()
    self.local = MLP(
      in_size, hidden_size,
      depth=mlp_depth,
      batch_norm=False,
      activation=activation
    )
    self.project_in = nn.Linear(in_size, out_size)
    self.interact = NeighbourDotMultiHeadAttention(
      hidden_size, out_size, attention_size,
      query_size=query_size, heads=heads
    )
    self.local_bn = nn.LayerNorm(hidden_size)
    self.residual_bn = nn.LayerNorm(out_size)
    self.activation = activation

  def forward(self, data, structure):
    local = self.activation(self.local_bn(self.local(data)))
    interaction = self.interact(local, local, structure)
    return self.residual_bn(interaction + self.project_in(data))
