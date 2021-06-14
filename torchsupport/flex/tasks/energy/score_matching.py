from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as func

from torch.autograd import grad

from torchsupport.data.io import make_differentiable
from torchsupport.data.namedtuple import namespace
from torchsupport.data.collate import default_collate

def _select(args, kwargs, names):
  results = []
  for name in names:
    if isinstance(name, int):
      results.append(args[name])
    else:
      results.append(kwargs[name])
  return results

def energy_score_aux(energy, grad_args, *args, **kwargs):
  grad_vars = _select(args, kwargs, grad_args)
  make_differentiable(grad_vars)
  E = energy(*args, **kwargs)
  score = grad(
    E, grad_vars,
    # E = -log p + C -> score = -grad E
    grad_outputs=-torch.ones_like(E),
    create_graph=True
  )
  if len(score) == 1:
    score = score[0]
  return score

def energy_score(energy, *grad_args):
  return partial(energy_score_aux, energy, grad_args)

def gaussian_noise(data, noise_level):
  # distribute over tuples
  if isinstance(data, (list, tuple)):
    return type(data)(
      gaussian_noise(item, noise_level)
      for item in data
    )
  if not torch.is_tensor(noise_level):
    return data + noise_level * torch.randn_like(data)
  assert noise_level.dim() == 1
  expand = (data.dim() - 1) * [1]
  noise_level = noise_level.to(data.device)
  noise_level = noise_level.view(noise_level.size(0), *expand)
  return data + noise_level * torch.randn_like(data)

def constant_noise_aux(level=0.1, batch_size=1):
  return level

def constant_noise(level=0.1):
  return partial(constant_noise_aux, level=level)

def functional_noise_aux(function, batch_size=1):
  return function(torch.rand(batch_size))

def functional_noise(function):
  return partial(functional_noise_aux, function)

def linear_noise(start=1e-6, stop=10):
  return functional_noise(lambda x: start + (stop - start) * x)

def geometric_noise(start=1e-3, stop=10):
  return functional_noise(lambda x: stop * (start / stop) ** x)

default_noise = constant_noise(level=1e-3)

def denoising_score(score, data, noised, noise_level):
  # distribute over tuples
  if isinstance(data, (list, tuple)):
    return sum(
      denoising_score(sc, d, n, noise_level)
      for sc, d, n in zip(score, data, noised)
    ) / len(data)
  return ((score + (noised - data) / noise_level ** 2) ** 2).mean()

def sliced_score(score, data):
  # distribute over tuples
  if isinstance(data, (list, tuple)):
    return sum(sliced_score(sc, d) for sc, d in zip(score, data))
  score = score.view(score.size(0), -1)
  v = torch.randn_like(score)
  grad_score = grad(score, data, grad_outputs=v, create_graph=True)[0]
  tr_grad_score = torch.einsum("ij,ij->i", grad_score, v)
  norm_score = (score ** 2).sum(dim=1)
  return (tr_grad_score + norm_score / 2).mean()

def finite_difference_input(data, eps=1e-3, energy=False, parallel=True):
  # distribute over tuples
  if isinstance(data, (list, tuple)):
    result = type(data)(
      finite_difference_input(item, eps=eps)
      for item in data
    )
    return [
      [item[idx] for item in result]
      for idx in range(len(result[0]))
    ]
  v = eps * torch.randn(data)
  versions = [data + v, data - v]
  if energy:
    versions = [data] + versions
  if parallel:
    data = torch.cat(versions, dim=0)
    return data, v
  return versions + [v]

def _replicate_aux(data, n=2):
  if isinstance(data, (list, tuple)):
    return [_replicate_aux(item, n=n) for item in data]
  return torch.cat(n * [data], dim=0)

def _split_aux(score, n=2):
  if isinstance(score, (list, tuple)):
    return [_split_aux(item, n=n) for item in score]
  return score.chunk(n, dim=0)

# TODO
def finite_difference_energy(E_0, E_p, E_m, v, eps=1e-3):
  # distribute over tuples
  if isinstance(E_0, (list, tuple)):
    return sum(
      finite_difference_energy(E_0_i, E_p_i, E_m_i, v_i, eps=eps)
      for E_0_i, E_p_i, E_m_i, v_i in zip(E_0, E_p, E_m, v)
    )
  E_0 = E_0.view(E_0.size(0), -1)
  E_p = E_p.view(E_p.size(0), -1)
  E_m = E_m.view(E_m.size(0), -1)
  tr_grad_score = 2 * E_0 - E_p - E_m
  norm_score = ((E_m - E_p) ** 2) / 8
  return (tr_grad_score.mean() + norm_score.mean()) / eps ** 2

def finite_difference_score(s_p, s_m, v, eps=1e-3):
  # distribute over tuples
  if isinstance(s_p, (list, tuple)):
    return sum(
      finite_difference_score(s_p_i, s_m_i, v_i, eps=eps)
      for s_p_i, s_m_i, v_i in zip(s_p, s_m, v)
    )
  s_p = s_p.view(s_p.size(0), -1)
  s_m = s_p.view(s_m.size(0), -1)
  norm_score = ((s_p + s_m) ** 2).sum(dim=1) / (8 * s_p.size(1))
  tr_grad_score = (v * s_p - v * s_m).sum(dim=1) / (2 * eps ** 2)
  return norm_score.mean() + tr_grad_score.mean()

def finite_difference_score_parallel(score, data, args, noise_level=0.0, eps=1e-3):
  if noise_level:
    data = gaussian_noise(data, noise_level)
  data, v = finite_difference_input(data, eps=eps)
  score_val = score(
    data,
    _replicate_aux(noise_level),
    *_replicate_aux(args)
  )
  s_p, s_m = _split_aux(score_val)
  loss = finite_difference_score(s_p, s_m, v, eps=eps)
  return loss, namespace(
    s_p=s_p, s_m=s_m, v=v
  )

def finite_difference_score_serial(score, data, args, noise_level=0.0, eps=1e-3):
  if noise_level:
    data = gaussian_noise(data, noise_level)
  data_p, data_m, v = finite_difference_input(data, eps=eps, parallel=False)
  s_p = score(data_p, noise_level, *args)
  s_m = score(data_m, noise_level, *args)
  loss = finite_difference_score(s_p, s_m, v, eps=eps)
  return loss, namespace(
    s_p=s_p, s_m=s_m, v=v
  )

def run_fd_score(score, data, args, noise_level=0.0, eps=1e-3, parallel=True):
  if parallel:
    return finite_difference_score_parallel(
      score, data, args, noise_level=noise_level, eps=eps
    )
  return finite_difference_score_serial(
    score, data, args, noise_level=noise_level, eps=eps
  )

def run_denoising_score(score, data, args, noise_level):
  noised = gaussian_noise(data, noise_level)
  score_val = score(noised, noise_level, *args)
  loss = denoising_score(score_val, data, noised, noise_level)
  return loss, namespace(
    data=data, noised=noised,
    score=score_val, noise_level=noise_level
  )

def run_sliced_score(score, data, args, noise_level=0.0):
  if noise_level:
    data = gaussian_noise(data, noise_level)
  make_differentiable(data)
  score_val = score(data, noise_level, *args)
  loss = sliced_score(score_val, data)
  return loss, namespace(
    data=data, score=score_val, noise_level=noise_level
  )

def score_matching_step(score, data, noise_source=None,
                        score_matching=run_denoising_score,
                        ctx=None):
  data, condition = data.sample(ctx.batch_size)
  noise = (noise_source or default_noise)(ctx.batch_size)
  loss, args = score_matching(score, data, condition, noise)
  ctx.argmin(score_loss=loss)
  return args
