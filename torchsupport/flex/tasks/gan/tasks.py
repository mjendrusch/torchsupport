from functools import partial
from torchsupport.flex.tasks.utils import parallel_steps

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.data.namedtuple import namespace
from torchsupport.flex.tasks.gan.losses import non_saturating

def run_discriminator(discriminator, real_data, fake_data,
                      gan_loss=non_saturating,
                      gan_loss_kwargs=None,
                      ctx=None):
  real, fake = discriminator(real_data), discriminator(fake_data)
  loss = gan_loss(ctx=ctx, **(gan_loss_kwargs or {})).critic(real, fake)
  return loss, namespace(
    real_data=real_data, fake_data=fake_data,
    real=real, fake=fake, ctx=ctx
  )

def discriminator_step(generator, discriminator, data,
                       gan_loss=non_saturating,
                       gan_loss_kwargs=None,
                       ctx=None):
  real_data = data.sample(ctx.batch_size)
  fake_data = generator.sample(ctx.batch_size)
  loss, args = run_discriminator(
    discriminator, real_data, fake_data,
    gan_loss=gan_loss, gan_loss_kwargs=gan_loss_kwargs,
    ctx=ctx
  )
  ctx.argmin(discriminator_loss=loss)
  return args

def run_generator(generator, discriminator,
                  gan_loss=non_saturating,
                  gan_loss_kwargs=None,
                  ctx=None):
  fake_data = generator.sample(ctx.batch_size)
  fake = discriminator(fake_data)
  loss = gan_loss(ctx=ctx, **gan_loss_kwargs).generator(fake)
  return loss, namespace(
    fake_data=fake_data, fake=fake, ctx=ctx
  )

def generator_step(generator, discriminator,
                   gan_loss=non_saturating,
                   gan_loss_kwargs=None,
                   ctx=None):
  loss, args = run_generator(
    generator, discriminator,
    gan_loss=gan_loss, gan_loss_kwargs=gan_loss_kwargs,
    ctx=ctx
  )
  ctx.argmin(generator_loss=loss)
  return args
