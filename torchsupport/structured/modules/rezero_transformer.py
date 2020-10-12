import torch
import torch.nn as nn
import torch.nn.functional as func

class MultiHeadAttention(nn.Module):  #definition of the class + where it inherits from
    r'''Implements MultiHeadAttention as described in the "Attention is all you need" paper. 
    Args: 
        in_size (int): input size
        n_heads (int): number of heads (for multihead attention)
        attention_size (int): number of features used to compare query and key in the 
          attention kernel
        value_size (int): value size (needed for defining value)
        out_size (int): output size 
    '''
  def __init__(self, in_size, n_heads, attention_size, value_size, out_size):  #attention_size=size for key and query, value - for value
    super().__init__()
    self.n_heads = n_heads
    self.attention_size = attention_size
    self.value_size = value_size
    self.in_size = in_size
    self.out_size = out_size

    self.value = nn.Linear(in_size, value_size * n_heads)
    self.key = nn.Linear(in_size, attention_size * n_heads)
    self.query = nn.Linear(in_size, attention_size * n_heads)
    self.out = nn.Linear(value_size * n_heads, out_size)

  def forward(self, inputs, index):
    key = self.key(inputs).view(inputs.size(0), self.n_heads , self.attention_size) #Create key matrix for input
    query = self.query(inputs).view(inputs.size(0), self.n_heads , self.attention_size)
    value = self.value(inputs).view(inputs.size(0), self.n_heads , self.value_size)

    key,_,indices,_ = pad(key, index) #pad the missing values in key (now all vectors have the same length)
    query,_, indices,_ = pad(query, index)
    value,_,indices,_ = pad(value, index)

    attention = dot_attention(key[:, :, None], query[:, None, :]).unsqeeze(-1) #creating the dot-attention product

    mask = torch.ones_like(attention, dtype=torch.bool) #returns a tensor filled with 1 with the same sie as the input - here attention
    mask.view(-1, *mask.shape[2:])[indices] = False  #soncatenate the heads in the mask and set all places with indeces to 0
    attention[mask] = -float('inf')
    attention = attention.softmax(dim=1) #make a softax over each dimension

    result = (value[:, :, None] * attention).sum(dim=1) #final result -> values multiplied by the attention and summed over in each dimension
    result = result.view(*result.shape[:-2], -1)
    result = unpad(result, indices)
    result = self.out(result)
    return result

class FeedForward(nn.Module):
    r'''Implements the Feed forward part of the transformer as described in
      the "Attention is all you need" paper. 
    Args: 
        in_size (int): input size
        hidden_size (int): output size for layers between the input and the output in 
          the feed forward network
        dropout (int): defines the dropout parameter
        depth (int): defines how many times the nn.Linear(hidden_size, hidden_size) is called
    '''
  def __init__(self, in_size, hidden_size, dropout=0.1, depth=2):
    super().__init__()
    self.in_size = in_size
    self.hidden_size = hidden_size
    self.dropout = nn.Dropout(dropout)

    self.preprocess = nn.Linear(in_size, hidden_size)
    self.original_name = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
      for _ in range(depth-2)])
    self.postprocess = nn.Linear(hidden_size, in_size)

  def forward(self, inputs):
    inputs = self.preprocess(inputs)
    for block in self.original_name:
      inputs = func.relu(block(inputs))
    inputs = self.dropout(inputs)
    inputs = self.postprocess(inputs)
    return inputs

class Normalization(nn.Module):
    r'''Implements the normalization term for the transformer as described in "Attention
      is all you need". 
    Args: 
        in_size (int): input size
        epsilon (int): correction of the division term (so that we don't divide over 0)
    '''
  def __init__(self, in_size, epsilon=1e-6):
    super().__init__()
    self.epsilon = epsilon
    self.in_size = in_size

    self.alpha = nn.Parameter(torch.ones(self.in_size))
    self.bias = nn.Parameter(torch.zeros(self.in_size))

  def forward(self, inputs):
    norm = self.alpha * (inputs - inputs.mean(dim=-1, keepdim=True))
      /(inputs.std(dim=-1, keepdim=True)+self.epsilon) + self.bias
    return norm

class StandardTransformer(nn.Module):
    r'''Implements brings all elements together to implement the transformer
      as described in "Attention is all you need". Uses the previously described functions:
      MultiHeadAttention, FeedForward and Normalization.
    Args: 
        in_size (int): input size
        n_heads (int): number of heads (for multiheadattention)
        hidden_size (int): output size for layers between the input and the output in 
          the feed forward network
        attention_size (int): number of features used to compare query and key in the 
          attention kernel
        value_size (int): value size (for multiheadattention)
        out_size (int): output size for the attention
        dropout (int): dropout parameter
    '''
  def __init__(self, in_size, n_heads, hidden_size,
               attention_size, value_size, out_size,
               dropout=0.1):
    super().__init__()
    self.norm_1 = Normalization(in_size)
    self.norm_2 = Normalization(in_size)

    self.attention = MultiHeadAttention(
      in_size, n_heads, attention_size, value_size, out_size
    )

    self.ff = FeedForward(in_size, hidden_size)

    self.dropout_1 = nn.Dropout(dropout)
    self.dropout_2 = nn.Dropout(dropout)

  def forward(self, inputs, index):
    inputs_norm = self.norm_1(inputs)
    inputs = inputs + self.dropout_1(
        self.attention(inputs_norm, index)
    )
    inputs_norm = self.norm_2(inputs)
    inputs = inputs + self.dropout_2(self.ff(inputs_norm))
    return inputs

class RezeroTransformer(ReZero):
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
  def __init__(self, in_size, n_heads, hidden_size,
               attention_size, value_size, out_size, 
               function, dropout=0.1):
    super()._init__()
    self.attention = MultiHeadAttention(
      in_size, n_heads, attention_size, value_size, out_size
    )
    self.ff = FeedForward(in_size, hidden_size)
    self.rezero = nn.Rezero(out_size)

    self.dropout_1 = nn.Dropout(dropout)
    self.dropout_2 = nn.Dropout(dropout)

def forward(self, inputs, index):
  result_1 = self.dropout_1(self.attention(inputs, index))
  inputs = self.rezero(inputs, result_1)
  result_2 = self.dropout_2(self.ff(inputs))
  inputs = self.rezero(inputs, result_2)
  return inputs



