import torch
import torch.nn as nn
import torch.nn.functional as func

class Adversarial(nn.Module):
  def __init__(self, generator, discriminator):
    self.generator = generator
    self.discriminator = discriminator

  def generate(self, *input):
    return self.generator(*input)

  def check(self, *check_input):
    return self.discriminator(*check_input)

def train(net, loss, optimizer, gen_inputs, real_inputs, setsize=10, epochs=1000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    batch_results = []
    batch_labels = []
    for idx in range(setsize):
      gen_input = next(gen_inputs)
      real_input = next(real_inputs)
      batch_results.append(net.generator(*gen_input).unsqueeze(0))
      batch_results.append(real_input)
      batch_labels.append(torch.tensor([[[0]]]))
      batch_labels.append(torch.tensor([[[1]]]))
    batch_tensor = torch.cat(batch_results, dim=0).to(device)
    label_tensor = torch.cat(batch_labels, dim=0).to(device)
    discriminator_output = net.discriminator(batch_tensor)
    loss_val = loss(batch_results, discriminator_output, label_tensor)
    loss_val.backwards()
    optimizer.step()
  return loss_val.item()