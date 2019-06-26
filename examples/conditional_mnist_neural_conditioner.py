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
    data, label_index = self.data[index]
    label = torch.zeros(10)
    label[label_index] = 1
    available = self.square_mask(data.size())
    requested = self.square_mask(data.size())
    available_label, requested_label = torch.randint(0, 2, (2,)).to(torch.float)
    return (
      (data, label),
      (available, available_label.unsqueeze(0)),
      (requested, requested_label.unsqueeze(0))
    )

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
    self.label_encoder = nn.Sequential(
      spectral_norm(nn.Linear(12, 16)),
      nn.ReLU(),
      spectral_norm(nn.Linear(16, 32)),
      nn.ReLU(),
      spectral_norm(nn.Linear(32, self.z)),
      nn.ReLU()
    )
    self.combinator = nn.Linear(2 * self.z, self.z)
    self.decoder = nn.Sequential(
      spectral_norm(nn.Linear(z, 128)),
      nn.ReLU(),
      spectral_norm(nn.Linear(128, 256)),
      nn.ReLU(),
      spectral_norm(nn.Linear(256, 28 * 28)),
      nn.Sigmoid()
    )
    self.label_decoder = nn.Sequential(
      spectral_norm(nn.Linear(self.z, 32)),
      nn.ReLU(),
      spectral_norm(nn.Linear(32, 16)),
      nn.ReLU(),
      spectral_norm(nn.Linear(16, 10)),
      nn.Softmax(dim=1)
    )

  def sample(self, batch_size):
    return torch.randn(batch_size, self.z)

  def forward(self, sample, restricted_inputs, available, requested):
    image, label = restricted_inputs
    a_image, a_label = available
    r_image, r_label = requested

    image_inputs = torch.cat((image, a_image, r_image), dim=1)
    image_inputs = image_inputs.view(image_inputs.size(0), -1)

    label_inputs = torch.cat((label, a_label, r_label), dim=1)
    label_inputs = label_inputs.view(label_inputs.size(0), -1)

    image_code = self.encoder(image_inputs)
    label_code = self.label_encoder(label_inputs)
    combined = torch.cat((image_code, label_code), dim=1)
    code = self.combinator(combined) + sample

    image_decoded = self.decoder(code).view(image.size(0), 1, 28, 28)
    label_decoded = self.label_decoder(code)

    return (image_decoded, label_decoded)

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
      nn.Linear(128, 32),
      nn.ReLU()
    )
    self.label_blocks = nn.Sequential(
      nn.Linear(22, 16),
      nn.ReLU(),
      nn.Linear(16, 32),
      nn.ReLU()
    )
    self.combinator = nn.Linear(64, 1)

  def forward(self, avail, reqst, a, r):
    g_image, g_label = avail
    image, label = reqst
    a_image, a_label = a
    r_image, r_label = r

    image_inputs = torch.cat((g_image, image, a_image, r_image), dim=1)
    image_inputs = image_inputs.view(image_inputs.size(0), -1)

    label_inputs = torch.cat((g_label, label, a_label, r_label), dim=1)
    label_inputs = label_inputs.view(label_inputs.size(0), -1)

    image_code = self.blocks(image_inputs)
    label_code = self.label_blocks(label_inputs)
    combined = torch.cat((image_code, label_code), dim=1)
    return self.combinator(combined)

class MNISTConditionerTraining(NeuralConditionerTraining):
  def restrict_inputs(self, data, mask):
    image, label = data
    imask, lmask = mask
    return (image * imask, label * lmask)

  def run_generator(self, data):
    ii, aa, rr = data
    inputs, _ = ii
    available, _ = aa
    requested, _ = rr

    _, generated, _, _ = super().run_generator(data)

    gen = generated[0][0].view(1, 28, 28)

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

    return ii, generated, aa, rr

if __name__ == "__main__":
  mnist = MNIST("examples/", download=False, transform=ToTensor())
  data = NCDataset(mnist)

  generator = Generator(z=32)
  discriminator = Discriminator()

  training = MNISTConditionerTraining(
    generator, discriminator, data,
    network_name="conditional-mnist-nc",
    device="cuda:0",
    n_critic=10,
    batch_size=64,
    max_epochs=1000
  )

  training.train()
