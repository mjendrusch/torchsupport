import torch
import torch.nn as nn
import torch.nn.functional as func

def sample(dist):
  value = dist.sample()
  logprob = dist.log_prob(value)
  return value, logprob

def straight_through(op):
  def _func(*inputs, **kwargs):
    *result, = op(*inputs, **kwargs)
    surrogate = inputs
    result = [
      r.detach() + s - s.detach()
      for (r, s) in zip(result, surrogate)
    ]
    if len(result) > 1:
      return result
    return result[0]
  return _func

def hard_distribution(distribution, logits, **kwargs):
  dist = distribution(logits=logits, **kwargs)
  surrogate = dist.rsample()
  result_index = surrogate.argmax(dim=1)
  result = torch.zeros_like(logits)
  result[torch.arange(0, result.size(0)), result_index] = 1.0
  result = result.detach()
  return result + surrogate - surrogate.detach()

def hard_one_hot(logits, temperature=0.1):
  return hard_distribution(
    torch.distributions.RelaxedOneHotCategorical,
    logits,
    temperature=temperature
  )

def hard_bernouilli(logits, temperature=0.1):
  return hard_distribution(
    torch.distributions.RelaxedBernoulli,
    logits,
    temperature=temperature
  )

def reinforce(op, sample=sample):
  def _func(*inputs, **kwargs):
    samples = []
    logprobs = []
    for distribution in inputs:
      s, l = sample(distribution)
      samples.append(s)
      logprobs.append(l)
    logprob = sum(logprobs)
    result = op(*samples, **kwargs).detach()
    surrogate = result * logprob
    return result + surrogate - surrogate.detach()
  return _func

class Reinforce(nn.Module):
  def __init__(self, target, sample=sample):
    super(Reinforce, self).__init__()
    self.target = reinforce(target, sample=sample)

  def forward(self, inputs):
    return self.target(inputs)

## TODO: implement Rebar, Relax, DiCE etc.
# class Lax(nn.Module):
#   def __init__(self, target, sample=sample, joint=False):
#     super(Lax, self).__init__()
#     self.target = target
#     self.sample = sample
#     self.joint = joint
#     self.value = ...

#   def value_loss(self, inputs=None, forward=None):
#     if inputs is not None:
#       result = self.forward(inputs)
#       loss = torch.autograd.grad(result, self.target.parameters(),
#                                  retain_graph=True, create_graph=True) ** 2
#       return loss
#     elif forward is not None:
#       return torch.autograd.grad(result, self.target.parameters(),
#                                  retain_graph=True, create_graph=True) ** 2
#     else:
#       raise ValueError("Either inputs or forward must be not None.")

#   def forward(self, inputs):
#     sample = self.sample(inputs)
#     result = (self.target(sample) - self.value(inputs)).detach() * torch.log(inputs)
#     result = result + self.value(sample)
#     if self.joint:
#       return result, self.value_loss(forward=result)
#     return result
