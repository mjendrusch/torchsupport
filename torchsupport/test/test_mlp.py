import pytest
import torch
from torch.nn.utils import spectral_norm
from torchsupport.modules import MLP

@pytest.mark.parametrize(
  'count, in_size, out_size, hidden_size, depth, batch_norm, normalization', [
    (count, in_size, out_size, hidden_size, depth, batch_norm, normalization)
    for count in (2, 10)
    for in_size in (1, 3)
    for out_size in (1, 4)
    for hidden_size in (1, 50, [10, 20, 30])
    for depth in (1, 3)
    for batch_norm in (True, False)
    for normalization in (lambda x: x, spectral_norm)
  ]
)
def test_mlp(count, in_size, out_size,
             hidden_size, depth, batch_norm,
             normalization):
  mlp = MLP(
    in_size, out_size,
    hidden_size=hidden_size,
    depth=depth,
    batch_norm=batch_norm,
    normalization=normalization
  )
  inputs = torch.randn(count, in_size)
  result = mlp(inputs)
  if isinstance(hidden_size, (list, tuple)):
    intermediate = inputs
    for idx, block in enumerate(mlp.blocks[:-1]):
      intermediate = block(intermediate)
      assert intermediate.size(1) == hidden_size[idx]
  else:
    intermediate = mlp.blocks[0](inputs)
    assert intermediate.size(1) == hidden_size
  assert result.size(1) == out_size
