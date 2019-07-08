import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.basic import MLP
from torchsupport.structured.modules.basic import NeighbourDotMultiHeadAttention

class Transformer(nn.Module):
  def __init__(self, in_size, out_size, hidden_size,
               query_size=None, attention_size=64,
               heads=8, mlp_depth=3, dropout=0.1,
               activation=func.elu_):
    super(Transformer, self).__init__()
    self.local = MLP(
      in_size, hidden_size,
      depth=mlp_depth,
      batch_norm=False,
      activation=activation
    )
    self.dropout = nn.Dropout(dropout, inplace=True)
    self.project_in = nn.Linear(in_size, out_size)
    self.project_down = nn.Linear(in_size, hidden_size)
    self.interact = NeighbourDotMultiHeadAttention(
      hidden_size, out_size, attention_size,
      query_size=query_size, heads=heads
    )
    self.local_bn = nn.LayerNorm(hidden_size)
    self.residual_bn = nn.LayerNorm(out_size)
    self.activation = activation

  def forward(self, data, structure):
    local = self.local(data) + self.project_down(data)
    local = self.dropout(local)
    local = self.activation(
      self.local_bn(local + self.project_down(data))
    )
    interaction = self.interact(local, local, structure)
    interaction = self.dropout(interaction)
    return self.residual_bn(interaction + self.project_in(data))
