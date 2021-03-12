from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import OneHotCategorical

from torchsupport.training.state import (
  NetNameListState, TrainingState
)
from torchsupport.training.multistep_training import MultistepTraining, step_descriptor
from torchsupport.data.io import to_device, make_differentiable
from torchsupport.data.collate import DataLoader, default_collate
from torchsupport.data.match import match

class HybridGenerativeTraining(MultistepTraining):
  def mixing_key(self, data):
    return data

  def run_encoder(self, *args):
    raise NotImplementedError("Abstract.")

  def encoder_loss(self, *args):
    raise NotImplementedError("Abstract.")

  @step_descriptor(n_steps="n_encoder", every="every_encoder")
  def encoder_step(self, data):
    args = self.run_encoder(data)
    return self.encoder_loss(*args)

  def run_decoder(self, *args):
    raise NotImplementedError("Abstract.")

  def decoder_loss(self, *args):
    raise NotImplementedError("Abstract.")

  @step_descriptor(n_steps="n_decoder", every="every_decoder")
  def decoder_step(self, data):
    args = self.run_decoder(data)
    return self.decoder_loss(*args)

  def run_prior(self, *args):
    raise NotImplementedError("Abstract.")

  def prior_loss(self, *args):
    raise NotImplementedError("Abstract.")

  @step_descriptor(n_steps="n_prior", every="every_prior")
  def prior_step(self, data):
    args = self.run_prior(data)
    return self.prior_loss(*args)

  def run_regularizer(self, *args):
    raise NotImplementedError("Abstract.")

  def regularizer_loss(self, *args):
    raise NotImplementedError("Abstract.")

  @step_descriptor(n_steps="n_regularizer", every="every_regularizer")
  def regularizer_step(self, data):
    args = self.run_regularizer(data)
    return self.regularizer_loss(*args)

  def run_classifier(self, *args):
    return ...

  def classifier_loss(self, *args):
    return None

  @step_descriptor(n_steps="n_classifier", every="every_classifier")
  def classifier_step(self, data):
    args = self.run_classifier(data)
    return self.classifier_loss(*args)

class HybridALAE(HybridGenerativeTraining):
  step_order = [
    "encoder_step", "decoder_step", "prior_step", "classifier_step", "regularizer_step"
  ]
  def __init__(self, encoder, decoder, prior, discriminator, classifier, data,
               classifier_data=None, optimizer=None, optimizer_kwargs=None,
               gamma=1.0, network_options=None, mapping_options=None,
               data_options=None, **kwargs):
    optimizer = optimizer or torch.optim.AdamW
    optimizer_kwargs = optimizer_kwargs or {}
    classifier_data = classifier_data or data
    network_options = network_options or {}
    mapping_options = mapping_options or {}
    data_options = data_options or {}

    self.encoder = ...
    self.decoder = ...
    self.prior = ...
    self.discriminator = ...
    self.classifier = ...

    networks = dict(
      encoder=(encoder, optimizer, optimizer_kwargs),
      decoder=(decoder, optimizer, optimizer_kwargs),
      prior=(prior, optimizer, optimizer_kwargs),
      discriminator=(discriminator, optimizer, optimizer_kwargs),
      classifier=(classifier, optimizer, optimizer_kwargs)
    )
    networks.update(network_options)

    mapping = dict(
      encoder_step=["encoder", "discriminator"],
      decoder_step=["prior", "decoder"],
      prior_step=["encoder", "decoder"],
      regularizer_step=["encoder", "discriminator"],
      classifier_step=["classifier"]
    )
    mapping.update(mapping_options)

    data = dict(
      encoder_step=data,
      decoder_step=None,
      prior_step=data,
      regularizer_step=data,
      classifier_step=classifier_data
    )
    data.update(data_options)

    super().__init__(networks, mapping, data, **kwargs)

    self.gamma = gamma

  def encode(self, data):
    z = self.encoder(data)
    y = self.classifier(data)
    return z, y

  def classify(self, data):
    return self.classifier(data)

  def each_generate(self, data):
    self.writer.add_images("generated", data.detach().cpu(), self.step_id)

  def run_encoder(self, data):
    codes = self.encode(data)
    real_result = self.discriminator(data, *codes)
    prior_codes = self.prior.sample(self.batch_size)
    generator_result = self.decoder(*prior_codes)
    # self.each_generate(generator_result)
    generated_codes = self.encode(generator_result)
    fake_result = self.discriminator(generator_result, *generated_codes)
    return real_result, fake_result

  def encoder_loss(self, real_result, fake_result):
    result = func.softplus(fake_result).mean() + func.softplus(-real_result).mean()
    self.current_losses["encoder"] = float(result)
    return result

  def run_decoder(self, data):
    prior_codes = self.prior.sample(self.batch_size)
    generator_result = self.decoder(*prior_codes)
    generated_codes = self.encode(generator_result)
    fake_result = self.discriminator(generator_result, *generated_codes)
    return (fake_result,)

  def decoder_loss(self, fake_result):
    result = func.softplus(-fake_result).mean()
    self.current_losses["decoder"] = float(result)
    return result

  def run_prior(self, data):
    real_codes = self.encode(data)
    prior_codes = self.prior.sample(self.batch_size)
    prior_codes = list(map(lambda x: torch.cat(x, dim=0), zip(real_codes, prior_codes)))
    generator_result = self.decoder(*prior_codes)
    generated_codes = self.encode(generator_result)
    reconstruction = self.decoder(*generated_codes)
    self.each_generate((generator_result, reconstruction))
    print((prior_codes[1].argmax(dim=1) == generated_codes[1].argmax(dim=1)).float().mean())
    return generated_codes, prior_codes

  def prior_loss(self, generated_codes, prior_codes):
    result = 0.0
    for idx, (c_generated, c_prior) in enumerate(zip(generated_codes, prior_codes)):
      this_loss = ((c_generated - c_prior) ** 2).mean()
      result = result + this_loss
      self.current_losses[f"prior {idx}"] = float(this_loss)
    # result = match(generated_codes, prior_codes).mean()
    self.current_losses["prior total"] = float(result)
    return result

  def run_regularizer(self, data):
    make_differentiable(data)
    codes = self.encode(data)
    discriminator_result = self.discriminator(data, *codes)
    gradient = torch.autograd.grad(
      discriminator_result, self.mixing_key(data),
      grad_outputs=torch.ones_like(discriminator_result),
      create_graph=True, retain_graph=True
    )[0]
    gradient = gradient.view(gradient.size(0), -1)
    gradient = (gradient ** 2).sum(dim=1)
    return (gradient,)

  def regularizer_loss(self, gradient):
    reg = self.gamma * gradient.mean() / 2
    self.current_losses["regularizer"] = float(reg)
    return reg

class HybridVAE(HybridALAE):
  def __init__(self, *args, mapping_options=None, **kwargs):
    mapping_options = mapping_options or {}
    mapping_options.update(dict(prior_step=["prior", "encoder", "decoder"]))
    super().__init__(*args, mapping_options=mapping_options, **kwargs)

  def run_prior(self, data):
    codes = self.encode(data)
    prior = self.prior()
    reconstruction = self.decoder(*codes)
    return data, reconstruction, prior, codes

  def prior_loss(self, data, reconstruction, prior, codes):
    rec_loss = match(data, reconstruction).mean()
    kl_loss = match(prior, codes).mean()
    self.current_losses["kullback leibler"] = float(kl_loss)
    self.current_losses["reconstruction"] = float(rec_loss)
    return rec_loss + kl_loss

class ClassifierALAE(HybridALAE):
  def run_classifier(self, data):
    data, labels = data
    prediction = self.classify(data)
    return prediction, labels

  def classifier_loss(self, prediction, labels):
    ce = func.cross_entropy(prediction, labels)
    self.current_losses["cross entropy"] = float(ce)
    return ce

class SharedEncoderALAE(ClassifierALAE):
  def classify(self, data):
    return self.encode(data)[1]

  def encode(self, data):
    z = self.encoder(data)
    y = self.classifier(z)
    return z, y

class ClassifierVAE(HybridVAE):
  def run_classifier(self, data):
    data, labels = data
    prediction = self.classify(data)
    return prediction, labels

  def classifier_loss(self, prediction, labels):
    ce = func.cross_entropy(prediction, labels)
    self.current_losses["cross entropy"] = float(ce)
    return ce

class SharedEncoderVAE(ClassifierVAE):
  def classify(self, data):
    return self.encode(data)[1]

  def encode(self, data):
    z = self.encoder(data)
    y = self.classifier(z)
    return z, y

