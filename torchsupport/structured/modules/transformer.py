import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.basic import MLP
from torchsupport.structured.modules.basic import NeighbourDotMultiHeadAttention
from torchsupport.structured import scatter

class UniversalTransformer(nn.Module):
  def __init__(self, in_size, out_size, hidden_size,
               query_size=None, attention_size=64,
               heads=8, dropout=0.1, activation=func.elu_):
    super(UniversalTransformer, self).__init__()
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

  def local(self, data, structure):
    raise NotImplementedError("Abstract.")

  def forward(self, data, structure):
    local = self.local(data, structure) + self.project_down(data)
    local = self.dropout(local)
    local = self.activation(
      self.local_bn(local + self.project_down(data))
    )
    interaction = self.interact(local, local, structure)
    interaction = self.dropout(interaction)

    # use pre-norm for stable training.
    return self.residual_bn(interaction) + self.project_in(data)

class Transformer(UniversalTransformer):
  def __init__(self, in_size, out_size, hidden_size,
               query_size=None, attention_size=64,
               heads=8, mlp_depth=3, dropout=0.1,
               activation=func.elu_):
    super(Transformer, self).__init__(
      in_size, out_size, hidden_size,
      query_size=query_size, attention_size=attention_size,
      heads=heads, dropout=dropout, activation=activation
    )
    self.local_block = MLP(
      in_size, hidden_size,
      depth=mlp_depth,
      batch_norm=False,
      activation=activation
    )

  def local(self, data, structure):
    return self.local_block(data)

class ConvTransformer(UniversalTransformer):
  def __init__(self, in_size, out_size, hidden_size,
               query_size=None, attention_size=64,
               heads=8, conv_kwargs=None, dropout=0.1,
               activation=func.elu_):
    super(ConvTransformer, self).__init__(
      in_size, out_size, hidden_size,
      query_size=query_size, attention_size=attention_size,
      heads=heads, dropout=dropout, activation=activation
    )
    self.local_block = nn.Conv1d(
      in_size, hidden_size, 3, **conv_kwargs
    )

  def local(self, data, structure):
    return scatter.batched(self.local_block, data, structure.indices)

class Halting(nn.Module):
  def __init__(self, module, size, threshold=0.5, max_iter=5):
    self.module = module
    self.probability = nn.Linear(size, 1)
    self.max_iter = max_iter
    self.threshold = threshold

  def forward(self, data, state, structure):
    probabilities = torch.zeros(state.size(0))
    remainders = torch.zeros(state.size(0))
    for _ in range(self.max_iter):
      p = self.probability(state).sigmoid()
      running = probabilities < 1.0
      new_halted = (probabilities + running.to(torch.float) * p) > self.threshold
      new_halted = new_halted * running
      running = (probabilities + running.to(torch.float) * p) <= self.threshold
      probabilities += p * running.to(torch.float)
      remainders += (1 - probabilities) * new_halted.to(torch.float)
      weights = p * running.to(torch.float) + remainders * new_halted.to(torch.float)
      new_state = self.module(data, state, structure)

      state = weights * new_state + (1 - weights) * state
      if not running.any():
        break
    
    return state
