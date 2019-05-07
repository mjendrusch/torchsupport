import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal, Categorical

def normal_kl_loss(mean, logvar):
  result = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp(), dim=0)
  return result.sum()

def normal_kl_norm_loss(mean, logvar, c=0.5,
                        keep_components=False):
  kld = normal_kl_loss(mean, logvar)
  result = torch.norm(kld - c, 1)
  if keep_components:
    return result, (kld,)
  return result

def gumbel_kl_loss(category):
  result = torch.sum(category * torch.log(category + 1e-20), dim=1)
  result = result.mean(dim=0)
  result += torch.log(torch.tensor(category.size(-1), dtype=result.dtype))
  return result

def gumbel_kl_norm_loss(category, c=0.5,
                        keep_components=False):
  kld = gumbel_kl_loss(category)
  result = torch.norm(kld - c, 1)
  if keep_components:
    return result, (kld,)
  return result

def vae_loss(parameters, reconstruction, target,
             keep_components=False):
  mean, logvar = parameters
  ce = func.binary_cross_entropy_with_logits(
    reconstruction, target,
    reduction='sum'
  ) / target.size(0)
  kld = normal_kl_loss(mean, logvar)
  result = ce + kld
  if keep_components:
    return result, (ce, kld)
  return result

def beta_vae_loss(parameters, reconstruction, target,
                  beta=20, c=0.5, keep_components=False):
  _, (ce, kld) = vae_loss(
    parameters, reconstruction, target,
    keep_components=True
  )
  norm_term = torch.norm(kld - c, 1)
  result = ce + beta * norm_term
  if keep_components:
    return result, (ce, kld, norm_term)
  return result

def categorical_vae_loss(parameters, reconstruction, target,
                         keep_components=False):
  if isinstance(parameters, (list, tuple)):
    category = parameters[0]
  else:
    category = parameters
  ce = func.binary_cross_entropy_with_logits(
    reconstruction, target,
    reduction='sum'
  ) / target.size(0)
  kld = gumbel_kl_loss(category)
  result = ce + kld
  if keep_components:
    return result, (ce, kld)
  return result

def categorical_beta_vae_loss(parameters, reconstruction, target,
                              beta=20, c=0.5, keep_components=False):
  _, (ce, kld) = categorical_vae_loss(parameters, reconstruction, target)
  norm_term = torch.norm(kld - c, 1)
  result = ce + beta * norm_term
  if keep_components:
    return result, (ce, kld, norm_term)
  return result

def joint_vae_loss(normal_parameters,
                   categorical_parameters,
                   reconstruction, target,
                   beta_normal=20, c_normal=0.5,
                   beta_categorical=20, c_categorical=0.5,
                   keep_components=False):
  result_normal, (ce, _, norm_term_normal) = beta_vae_loss(
    normal_parameters, reconstruction, target,
    beta=beta_normal, c=c_normal,
    keep_components=True
  )
  result_categorical, (norm_term_categorical,) = gumbel_kl_norm_loss(
    categorical_parameters, c=c_categorical, keep_components=True
  )
  result_categorical *= beta_categorical
  result = result_normal + result_categorical
  if keep_components:
    return result, (ce, norm_term_normal, norm_term_categorical)
  return result

def tc_encoder_loss(discriminator, true_sample):
  sample_prediction = discriminator(true_sample)
  return (sample_prediction[:, 0] - sample_prediction[:, 1]).mean()

def tc_discriminator_loss(discriminator, true_batch, shuffle_batch):
  shuffle_indices = [
    shuffle_batch[torch.randperm(shuffle_batch.size(0)), idx:idx+1]
    for idx in range(shuffle_batch.size(-1))
  ]
  shuffle_batch = torch.cat(shuffle_indices, dim=1)

  sample_prediction = discriminator(true_batch)
  shuffle_prediction = discriminator(shuffle_batch)

  softmax_sample = torch.softmax(sample_prediction, dim=1)
  softmax_shuffle = torch.softmax(shuffle_prediction, dim=1)

  discriminator_loss = \
    -0.5 * (torch.log(softmax_sample[:, 0]).mean() \
    + torch.log(softmax_shuffle[:, 1]).mean())

  return discriminator_loss

def factor_vae_loss(normal_parameters, tc_parameters,
                    reconstruction, target,
                    gamma=100, keep_components=False):
  vae, (ce, kld) = vae_loss(
    normal_parameters, reconstruction, target
  )
  tc_loss = tc_discriminator_loss(*tc_parameters)
  result = vae + gamma * tc_loss
  if keep_components:
    return result, (ce, kld, tc_loss)
  return result
