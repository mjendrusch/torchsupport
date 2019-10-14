import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from torchsupport.training.neural_conditioner import NeuralConditionerTraining

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

class NCDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def square_mask(self, size):
    _, w, h = size
    min_size = min(w, h)
    mask_size, = torch.randint(min_size // 2, 2 * min_size // 3, (1,))
    mask = torch.zeros(1, w, h)
    x, = torch.randint(0, w - mask_size + 1, (1,))
    y, = torch.randint(0, h - mask_size + 1, (1,))

    mask[:, x:x + mask_size, y:y + mask_size] = 1
    return mask

  def __getitem__(self, index):
    data, label = self.data[index]
    available = self.square_mask(data.size())
    requested = self.square_mask(data.size())
    return data, available, requested

  def __len__(self):
    return len(self.data)

class Generator(nn.Module):
  def __init__(self, z=32):
    super(Generator, self).__init__()
    self.z = z
    self.encoder = nn.Sequential(
      spectral_norm(nn.Linear(3 * 28 * 28, 256)),
      nn.ReLU(),
      spectral_norm(nn.Linear(256, 128)),
      nn.ReLU(),
      spectral_norm(nn.Linear(128, z)),
      nn.ReLU()
    )
    self.decoder = nn.Sequential(
      nn.Linear(z, 128),
      nn.ReLU(),
      nn.Linear(128, 256),
      nn.ReLU(),
      nn.Linear(256, 28 * 28),
      nn.Sigmoid()
    )

  def sample(self, batch_size):
    return torch.randn(batch_size, self.z)

  def forward(self, sample, restricted_inputs, available, requested):
    inputs = torch.cat((restricted_inputs, available, requested), dim=1)
    inputs = inputs.view(inputs.size(0), -1)
    return self.decoder(self.encoder(inputs) + sample).view(inputs.size(0), 1, 28, 28)

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.blocks = nn.Sequential(
      nn.Linear(4 * 28 * 28, 512),
      nn.ReLU(),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 1)
    )

  def forward(self, avail, reqst, a, r):
    inputs = torch.cat((avail, reqst, a, r), dim=1)
    inputs = inputs.view(inputs.size(0), -1)
    return self.blocks(inputs)

class MNISTConditionerTraining(NeuralConditionerTraining):
  def run_generator(self, data):
    inputs, available, requested = data
    _, generated, _, _ = super().run_generator(data)

    gen = generated[0].view(1, 28, 28)

    self.writer.add_image("inputs", normalize(inputs[0]), self.step_id)
    self.writer.add_image("available", normalize(inputs[0] * available[0]), self.step_id)
    self.writer.add_image("requested", normalize(inputs[0] * requested[0]), self.step_id)
    self.writer.add_image("generated", normalize(gen), self.step_id)
    self.writer.add_image(
      "generated request",
      normalize(gen * requested[0]),
      self.step_id
    )
    self.writer.add_image(
      "combined",
      normalize(
        inputs[0] * available[0] + gen * requested[0]
      ),
      self.step_id
    )

    return inputs, generated, available, requested

if __name__ == "__main__":
  mnist = MNIST("examples/", download=False, transform=ToTensor())
  data = NCDataset(mnist)

  generator = Generator(z=32)
  discriminator = Discriminator()

  training = MNISTConditionerTraining(
    generator, discriminator, data,
    network_name="mnist-nc",
    device="cpu",
    batch_size=64,
    max_epochs=1000
  )

  training.train()
