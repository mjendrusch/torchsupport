import torch
import torch.nn as nn
import torch.nn.functional as func
import torchsupport.distributions as dist

def sample(dist):
  value = dist.sample()
  logprob = dist.log_prob(value)
  return value, logprob

def rsample(dist):
  value = dist.rsample()
  hard_value = dist.harden(value)
  logprob = dist.log_prob(value)
  hard_logprob = dist.hard_distribution().log_prob(hard_value)
  return value, hard_value, logprob, hard_logprob

def replace_gradient(value, gradient_provider):
  r"""Replaces the gradient of one expression with the gradient
  of another.

  Args:
    value (torch.Tensor): tensor providing the value
      of the result.
    gradient_provider (torch.Tensor): tensor providing
      the autograd graph for gradient computation for
      the result.

  Returns:
    Tensor with numerical value given by value and gradients
    computed according to the graph of gradient_provider.

  .. warning::
      replacing gradient computation for a tensor with an
      inappropriate graph may do strange things to your trainig.
      Mind your gradients and know what you are doing.
  """
  result = value.detach()
  zero = gradient_provider - gradient_provider.detach()
  zero = zero.clamp_min(0.0)
  return result + zero

def straight_through(op):
  r"""Replaces gradients of an operation by the gradients of
  its inputs (straight-through estimator).

  Args:
    op (function): operation for gradient estimation.

  Returns:
    Operation with gradient estimation by straight-through
    estimator.
  """
  def _func(*inputs, **kwargs):
    *result, = op(*inputs, **kwargs)
    surrogate = inputs
    result = [
      replace_gradient(r, s)
      for (r, s) in zip(result, surrogate)
    ]
    if len(result) > 1:
      return result
    return result[0]
  return _func

def hard_distribution(distribution, logits, **kwargs):
  r"""Hardens the output of a reparameterizable distribution
  to provide hard samples with correct handling of gradient
  estimation.

  Args:
    distribution (torch.distributions.Distribution):
      distribution for gradient-aware hard sampling.
    logits (torch.Tensor): unnormalized log-likelihood tensor
      for the distribution.
    kwargs (dict): keyword arguments forwarded
      to the distribution.

  Returns:
    Hardened distribution with proper gradient handling.
  """
  dist = distribution(logits=logits, **kwargs)
  surrogate = dist.rsample()
  result = dist.harden(surrogate)
  return replace_gradient(result, surrogate)

def hard_one_hot(logits, temperature=0.1):
  r"""Categorical distribution with proper gradient estimation.

  For more details, refer to
    :class:`torch.distributions.RelaxedOneHotCategorical`.

  Args:
    logits (torch.Tensor): unnormalized log-likelihood tensor
      for the distribution.
    temperature (float): temperature of the distribution.

  Returns:
    Categorical distribution with proper gradient handling.
  """
  return hard_distribution(
    torch.distributions.RelaxedOneHotCategorical,
    logits,
    temperature=temperature
  )

def hard_bernouilli(logits, temperature=0.1):
  r"""Bernouilli distribution with proper gradient estimation.

  For more details, refer to
    :class:`torch.distributions.RelaxedBernouilli`.

  Args:
    logits (torch.Tensor): unnormalized log-likelihood tensor
      for the distribution.
    temperature (float): temperature of the distribution.

  Returns:
    Bernouilli distribution with proper gradient handling.
  """
  return hard_distribution(
    torch.distributions.RelaxedBernoulli,
    logits,
    temperature=temperature
  )

def _k_hot_sample_batch(weights):
  uniform = torch.rand_like(weights)
  gumbel = -torch.log(-torch.log(uniform + 1e-20) + 1e-20) + torch.log(weights + 1e-20)
  return gumbel

def _k_hot_compute_probability(alpha, temperature=0.1):
  alpha_linear = alpha.view(alpha.size(0), alpha.size(1), -1)
  result = torch.softmax(alpha_linear / temperature, dim=-1)
  return result.view(*alpha.size())

def _k_hot_relaxed_top_k(batch, k=1, temperature=0.1):
  alpha_idx = batch
  probability = _k_hot_compute_probability(alpha_idx, temperature=temperature)
  result = probability
  for _ in range(1, k):
    alpha_idx = alpha_idx + torch.log(1 - probability + 1e-20)
    probability = _k_hot_compute_probability(alpha_idx, temperature=temperature)
    result = result + probability
  return result

def soft_k_hot(logits, k, temperature=0.1):
  r"""Returns a soft k-hot sample given a categorical
  distribution defined by a tensor of unnormalized
  log-likelihoods.

  This is useful for example to sample a set of pixels in an
  image to move from a grid-structured data representation to a
  set- or graph-structured representation within a network.

  Args:
    logits (torch.Tensor): unnormalized log-likelihood tensor.
    k (int): number of items to sample without replacement.
    temperature (float): temparature of the soft distribution.

  Returns:
    Soft k-hot vector from the relaxed k-hot distribution
    defined by logits and temperature.
  """
  weights = torch.sigmoid(logits)
  batch = _k_hot_sample_batch(weights)
  top_k = _k_hot_relaxed_top_k(batch, k=k, temperature=temperature)
  return top_k

def hard_k_hot(logits, k, temperature=0.1):
  r"""Returns a hard k-hot sample given a categorical
  distribution defined by a tensor of unnormalized
  log-likelihoods.

  This is useful for example to sample a set of pixels in an
  image to move from a grid-structured data representation to a
  set- or graph-structured representation within a network.

  Args:
    logits (torch.Tensor): unnormalized log-likelihood tensor.
    k (int): number of items to sample without replacement.
    temperature (float): temparature of the soft distribution.

  Returns:
    Hard k-hot vector from the relaxed k-hot distribution
    defined by logits and temperature.
  """
  soft = soft_k_hot(logits, k, temperature=temperature)
  hard = torch.zeros_like(soft)
  _, top_k = torch.topk(logits, k)
  index = torch.repeat_interleave(torch.arange(0, hard.size(0)), k)
  hard[index, top_k.view(-1)] = 1.0
  return replace_gradient(hard, soft)

def reinforce(op, sample=sample):
  r"""Attaches a REINFORCE gradient estimator to
  an operation on samples from a given distribution.

  Args:
    op (function): operation to propagate gradients
      using REINFORCE.
    sample (function): sampling operation turning inputs
      into samples from a non-reparameterizable distribution.

  Returns:
    Operation :math:`\lambda x. op(sample(x))` with gradient
    estimation given by REINFORCE.
  """
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
    return replace_gradient(result, surrogate)
  return _func

class Reinforce(nn.Module):
  r"""Attaches a REINFORCE gradient estimator to
  an operation on samples from a given distribution.

  Args:
    target (function): operation to propagate gradients
      using REINFORCE.
    sample (function): sampling operation turning inputs
      into samples from a non-reparameterizable distribution.
  """
  def __init__(self, target, sample=sample):
    super(Reinforce, self).__init__()
    self.target = reinforce(target, sample=sample)

  def forward(self, inputs):
    return self.target(inputs)

## TODO: implement Rebar, Relax, DiCE etc.
class Lax(nn.Module):
  def __init__(self, target, value, sample=rsample, joint=False):
    super(Lax, self).__init__()
    self.target = target
    self.sample = sample
    self.joint = joint
    self.value = value

  def value_loss(self, inputs=None, forward=None):
    if inputs is not None:
      result = self.forward(inputs)
      loss = torch.autograd.grad(result, self.target.parameters(),
                                 retain_graph=True, create_graph=True) ** 2
      return loss
    elif forward is not None:
      return torch.autograd.grad(forward, self.target.parameters(),
                                 retain_graph=True, create_graph=True) ** 2
    else:
      raise ValueError("Either inputs or forward must be not None.")

  def forward(self, inputs):
    sample, hard, logprob, hard_logprob = self.sample(inputs)
    result = self.target(hard)
    control = self.value(sample)
    surrogate = result.detach() * hard_logprob - control.detach() * logprob + control
    result = replace_gradient(result, surrogate)
    if self.joint:
      return result, self.value_loss(forward=result)
    return result

class Relax(Lax):
  def __init__(self, target, value, sample=rsample, joint=False):
    super(Relax, self).__init__(
      target, value,
      sample=sample,
      joint=joint
    )

  def forward(self, inputs):
    sample, hard_sample, logprob, hard_logprob = self.sample(inputs)
    conditional_sample = inputs.conditional_sample(hard_sample)
    result = self.target(hard_sample)
    control = self.value(sample)
    conditional_control = self.value(conditional_sample)
    surrogate = (result.detach() - conditional_control.detach()) * hard_logprob
    surrogate += control - conditional_control
    result = replace_gradient(result, surrogate)
    if self.joint:
      return result, self.value_loss(forward=result)
    return result
