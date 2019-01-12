import torch
import torch.nn as nn
import torch.nn.functional as func

def training_loop_wnet(data, valid_data, encoder, decoder, optimizer,
                       regularization, reconstruction_loss,
                       validate_every=10, checkpoint_every=10,
                       report_every=10, board_prefix=pwd()):
  valid_regularization_value = None
  valid_loss_value = None
  for epoch in range(num_epochs):
    for idx, batch in enumerate(data):
      # step 1
      optimizer.zero_grad()
      encoder_out = encoder(batch)
      loss = regularization(encoder_out)
      loss.backward()
      optimizer.step()

      regularization_value = loss.item()

      # step 2
      optimizer.zero_grad()
      decoder_out = decoder(encoder_out)
      loss = reconstruction_loss(decoder_out, batch)
      loss.backward()
      optimizer.step()

      loss_value = loss.item()

      if idx % validate_every == 0:
        valid_batch = next(valid_data)
        out = encoder(valid_batch)
        valid_regularization_value = regularization(out).item()
        out = decoder(out)
        valid_loss_value = reconstruction_loss(out, batch).item()

      if idx % checkpoint_every == 0:
        ... # TODO

      if idx % report_every == 0:
        ... # TODO

  return encoder, decoder

def training_loop_residual_reconstruction(data, encoder, optimizer, board_prefix=pwd()):
  pass # TODO

def training_loop_multi_decoder(data, encoder, decoder, optimizer, board_prefix=pwd()):
  pass # TODO