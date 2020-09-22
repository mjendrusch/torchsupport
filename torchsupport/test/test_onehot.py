import pytest
import torch
from torch.nn.utils import spectral_norm
from torchsupport.modules import one_hot_encode, OneHotEncoder

CASES = [
  case
  for numeric in (True, False)
  for case in [
    (
      "ABCAABAC", "ABC",
      torch.tensor([0, 1, 2, 0, 0, 1, 0, 2]),
      numeric
    ),
    (
      list("ABCAABAC"), "ABC",
      torch.tensor([0, 1, 2, 0, 0, 1, 0, 2]),
      numeric
    ),
    (
      torch.tensor([0, 1, 2, 0, 0, 1, 0, 2]), [0, 1, 2],
      torch.tensor([0, 1, 2, 0, 0, 1, 0, 2]),
      numeric
    )
  ]
]

@pytest.mark.parametrize(
  "data, code, expected, numeric", CASES
)
def test_one_hot_encode_shape(data, code, expected, numeric):
  encoding = one_hot_encode(data, code, numeric=numeric)
  if numeric:
    assert encoding.dim() == 1
    assert encoding.size(0) == len(data)
  else:
    assert encoding.dim() == 2
    assert encoding.size(0) == len(code)
    assert encoding.size(1) == len(data)

@pytest.mark.parametrize(
  "data, code, expected, numeric", CASES
)
def test_one_hot_encode_value(data, code, expected, numeric):
  encoding = one_hot_encode(data, code, numeric=numeric)
  if numeric:
    assert bool((encoding == expected).all())
  else:
    expected_one_hot = torch.zeros(expected.size(0), len(code))
    ind = torch.arange(expected.size(0))
    expected_one_hot[ind, expected] = 1
    assert bool((encoding == expected_one_hot.t()).all())

@pytest.mark.parametrize(
  "data, code, expected, numeric", CASES
)
def test_one_hot_encode_consistent(data, code, expected, numeric):
  encoding = one_hot_encode(data, code, numeric=numeric)
  if not numeric:
    encoding = encoding.argmax(dim=0)
  decoding = []
  for base, encoded in zip(data, encoding):
    decoded = code[int(encoded)]
    decoding.append(decoded)
    assert decoded == base

  # check consistency
  re_encoding = one_hot_encode(decoding, code, numeric=True)
  print(re_encoding)
  assert bool((encoding == re_encoding).all())

def test_create_encoder():
  OneHotEncoder("ABC", numeric=True)
  OneHotEncoder("ABC", numeric=False)
  OneHotEncoder(list("ABC"))
  OneHotEncoder(torch.arange(10, dtype=torch.long))
