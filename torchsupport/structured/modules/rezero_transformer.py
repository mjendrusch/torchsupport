import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.basic import MLP
from torchsupport.modules.rezero import ReZero
from torchsupport.structured.modules.sequence_transformer import SequenceMultiHeadAttention

class RezeroTransformerBlock(nn.Module):
  r'''Implements a ReZero transformer as described in the paper:
    https://arxiv.org/pdf/2003.04887.pdf. Thereby here the normalization step is completely
    skipped and the multiheadattenition result and the feed forward result from the standard
    transformer are multiplied with the trainable alpha-parameter (desribed in rezero.py).
  Args:
    in_size (int): input size
    n_heads (int): number of heads (for multihead attention)
    hidden_size (int): output size for layers between the input and the output in
      the feed forward network
    attention_size (int): number of features used to compare query and key in the
      attention kernel
    value_size (int): value size (for multihead attention)
    out_size (int): output size (for multiheadatention)
    function (callable): Conv1d/Conv2D/Conv3D
    dropout (int): dropout parameter
  '''
  def __init__(self, size, n_heads=8, hidden_size=128,
               attention_size=128, value_size=128, depth=2, dropout=0.1):
    super().__init__()
    self.attention = SequenceMultiHeadAttention(
      size, size,
      attention_size=attention_size,
      hidden_size=value_size,
      heads=n_heads
    )
    self.ff = MLP(size, size, hidden_size=hidden_size, depth=2, batch_norm=False)
    self.rezero = ReZero(size)

    self.dropout_1 = nn.Dropout(dropout)
    self.dropout_2 = nn.Dropout(dropout)

  def forward(self, inputs, index):
    result_1 = self.dropout_1(self.attention(inputs, index))
    inputs = self.rezero(inputs, result_1)
    result_2 = self.dropout_2(self.ff(inputs))
    inputs = self.rezero(inputs, result_2)
    return inputs
