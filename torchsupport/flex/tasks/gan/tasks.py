import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.flex.tasks.gan.losses import non_saturating

def discriminator_step(generator, discriminator, data,
                       gan_loss=non_saturating,
                       gan_loss_kwargs=None,
                       regularization=None, ctx=None):
  real_data = data.sample(ctx.batch_size)
  fake_data = generator.sample(ctx.batch_size)
  real, fake = discriminator(real_data), discriminator(fake_data)
  loss = gan_loss(ctx=ctx, **gan_loss_kwargs).critic(real, fake)
  ctx.argmin(discriminator_loss=loss)
  regularization(
    real_data=real_data, fake_data=fake_data,
    real=real, fake=fake, ctx=ctx
  )

def generator_step(generator, discriminator,
                   gan_loss=non_saturating,
                   gan_loss_kwargs=None,
                   regularization=None, ctx=None):
  fake_data = generator.sample(ctx.batch_size)
  fake = discriminator(fake_data)
  loss = gan_loss(ctx=ctx, **gan_loss_kwargs).generator(fake)
  ctx.argmin(generator_loss=loss)
  regularization(
    fake_data=fake_data, fake=fake, ctx=ctx
  )
