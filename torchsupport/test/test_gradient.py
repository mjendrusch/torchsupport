import pytest
import torch
from torchsupport.modules import (
  replace_gradient, reinforce, straight_through
)

@pytest.fixture
def grad_sink():
  return torch.ones(10, requires_grad=True)

def test_replace_gradient(grad_sink):
  gradient_provider = grad_sink * 42
  replacement_value = torch.zeros_like(grad_sink)
  result = replace_gradient(replacement_value, gradient_provider)
  assert bool((result == replacement_value).all())
  replaced_grad = torch.autograd.grad(
    result, grad_sink,
    grad_outputs=torch.ones_like(result),
    retain_graph=True
  )[0]
  desired_grad = torch.autograd.grad(
    gradient_provider, grad_sink,
    grad_outputs=torch.ones_like(gradient_provider),
    retain_graph=True
  )[0]
  assert bool((replaced_grad == desired_grad).all())

def test_reinforce(grad_sink):
  grad_sink = grad_sink[None].expand(10, 10)
  op = lambda x: x.sum(dim=1, keepdim=True)
  reinforced_sum = reinforce(op)
  dist = torch.distributions.Normal(grad_sink, 0.1 * torch.ones_like(grad_sink))
  reinforce_dist = torch.distributions.Normal(grad_sink, 0.1)
  torch.random.manual_seed(1234)
  reinforced_result = reinforced_sum(reinforce_dist).mean(dim=0)
  reparam_result = dist.rsample().sum(dim=1).mean(dim=0)

  assert torch.allclose(
    reinforced_result[0], reparam_result,
    rtol=0.1, atol=0.1
  )

  # replaced_grad = torch.autograd.grad(
  #   reinforced_result, grad_sink,
  #   grad_outputs=torch.ones_like(reinforced_result),
  #   retain_graph=True
  # )[0]
  # desired_grad = torch.autograd.grad(
  #   reparam_result, grad_sink,
  #   grad_outputs=torch.ones_like(reparam_result),
  #   retain_graph=True
  # )[0]
  # assert torch.allclose(
  #   replaced_grad, desired_grad,
  #   rtol=0.1, atol=0.1
  # )
