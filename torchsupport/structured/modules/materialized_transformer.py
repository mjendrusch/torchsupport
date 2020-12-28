import torch
import torch.nn as nn

from torchsupport.modules import ReZero

class MaterializedMultiHeadAttention(nn.Module):
  def __init__(self, node_in_size, node_out_size, edge_in_size, edge_out_size,
               attention_size=32, heads=8, value_size=32):
    r"""Self-attention with materialized attention maps and "edge-features" on these
    attention maps. Warning: Implementing this on a hunch after seeing the AlphaFold
    blogpost - this may not train or make sense at all.

    Args:
      node_in_size (int): number of sequence feature maps. These corresponds to
        "node features" if one interprets an attention map as a soft adjacency matrix.
      node_out_size (int): number of sequence output feature maps.
      edge_in_size (int): number of feature maps of a materialized attention map.
        Interpreting the attention map as a soft adjacency matrix, its entries correspond
        to edges, to which we can attach learned edge features.
      edge_out_size (int): number of output feature maps of a materialized attention map.
      attention_size (int): size of vectors compared in dot-product attention.
      heads (int): number of attention heads.
      value_size (int): size of the value embedding.
    """
    super().__init__()
    self.heads = heads
    self.attention_size = attention_size
    self.query = nn.Conv1d(node_in_size, attention_size * heads, 1, bias=False)
    self.key = nn.Conv2d(node_in_size + edge_in_size, attention_size * heads, 1, bias=False)
    self.value = nn.Conv2d(node_in_size + edge_in_size, value_size * heads, 1, bias=False)
    self.out = nn.Conv1d(value_size * heads, node_out_size, 1, bias=False)
    self.edge_out = nn.Conv2d(value_size * heads + edge_in_size, edge_out_size, 1, bias=False)

  def forward(self, nodes, edges, mask):
    query = self.query(nodes)[:, :, :, None]
    node_edges = torch.cat((nodes[:, :, :, None].expand(*nodes.shape, edges.shape[-1]), edges), dim=1)
    key = self.key(node_edges)
    value = self.value(node_edges)
    value = value.view(value.size(0), self.heads, -1, *value.shape[2:])

    sim = (query * key).view(key.size(0), self.heads, self.attention_size, *key.shape[2:])
    sim = sim.sum(dim=2) / torch.tensor(self.attention_size, dtype=torch.float).sqrt()

    mm = mask[:, None, None, :]
    sim[~mm.expand_as(sim)] = -float("inf")
    sim = sim.softmax(dim=-1)
    
    mm = mask[:, None, None, :] * mask[:, None, :, None]
    sim = sim * mm.expand_as(sim).float()

    value = value * mm[:, None].expand_as(value).float()
    node_features = (sim[:, :, None] * value).sum(dim=-1)
    node_out = self.out(node_features.view(query.size(0), -1, query.shape[2]))

    edge_features = node_features.unsqueeze(-1).repeat_interleave(key.size(-1), dim=-1)
    edge_features = edge_features.view(key.size(0), -1, *key.shape[2:])
    edge_features_t = edge_features.transpose(-1, -2)
    edge_features = torch.cat(((edge_features + edge_features_t) / 2, edges), dim=1)
    edge_out = self.edge_out(edge_features)

    return node_out, edge_out

class MaterializedTransformerBlock(nn.Module):
  def __init__(self, node_in_size, node_out_size, edge_in_size, edge_out_size,
               attention_size=32, heads=8, value_size=32, dropout=0.1,
               kernel_size=1, dilation=1, activation=nn.ReLU()):
    r"""Transformer block with materialized attention maps and "edge-features" on these
    attention maps. Warning: Implementing this on a hunch after seeing the AlphaFold
    blogpost - this may not train or make sense at all.

    Args:
      node_in_size (int): number of sequence feature maps. These corresponds to
        "node features" if one interprets an attention map as a soft adjacency matrix.
      node_out_size (int): number of sequence output feature maps.
      edge_in_size (int): number of feature maps of a materialized attention map.
        Interpreting the attention map as a soft adjacency matrix, its entries correspond
        to edges, to which we can attach learned edge features.
      edge_out_size (int): number of output feature maps of a materialized attention map.
      attention_size (int): size of vectors compared in dot-product attention.
      heads (int): number of attention heads.
      value_size (int): size of the value embedding.
      kernel_size (int): kernel size of the local block.
      dilation (int): dilation of the local block, if applicable.
      dropout (float): dropout of the transformer block.
      activation (nn.Module): nonlinear activation function. Defaults to ReLU.
    """
    super().__init__()
    padding = kernel_size // 2 * dilation
    self.dropout = nn.Dropout(dropout)
    self.attention = MaterializedMultiHeadAttention(
      node_in_size, node_out_size, edge_in_size, edge_out_size,
      attention_size=attention_size, heads=heads, value_size=value_size
    )
    if node_in_size != node_out_size:
      self.project_node = nn.Conv1d(node_in_size, node_out_size, 1, bias=False)
    else:
      self.project_node = nn.Identity()
    if edge_in_size != edge_out_size:
      self.project_edge = nn.Conv2d(edge_in_size, edge_out_size, 1, bias=False)
    else:
      self.project_edge = nn.Identity()
    self.zero_node = nn.ModuleList([ReZero(node_out_size), ReZero(node_out_size)])
    self.zero_edge = nn.ModuleList([ReZero(edge_out_size), ReZero(edge_out_size)])

    self.node_mlp = nn.Sequential(
      nn.Conv1d(
        node_in_size, node_in_size, kernel_size,
        padding=padding, dilation=dilation
      ),
      activation,
      nn.Conv1d(
        node_in_size, node_in_size, kernel_size,
        padding=padding, dilation=dilation
      ),
      activation
    )
    self.edge_mlp = nn.Sequential(
      nn.Conv2d(
        edge_in_size, edge_in_size, kernel_size,
        padding=padding, dilation=dilation
      ),
      activation,
      nn.Conv2d(
        edge_in_size, edge_in_size, kernel_size,
        padding=padding, dilation=dilation
      ),
      activation
    )

  def forward(self, nodes, edges, mask):
    nodes = self.dropout(self.zero_node[0](nodes, self.node_mlp(nodes)))
    edges = self.dropout(self.zero_edge[0](edges, self.edge_mlp(edges)))
    node_features, edge_features = self.attention(nodes, edges, mask)
    nodes = self.dropout(self.zero_node[1](self.project_node(nodes), node_features))
    edges = self.dropout(self.zero_edge[1](self.project_edge(edges), edge_features))

    mask = mask[:, None]
    nodes[~mask.expand_as(nodes)] = 0.0
    mask = mask[:, :, None, :] * mask[:, :, :, None]
    edges[~mask.expand_as(edges)] = 0.0
    return nodes, edges
