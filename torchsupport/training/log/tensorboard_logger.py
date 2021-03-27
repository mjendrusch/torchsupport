from tensorboardX import SummaryWriter
from torchsupport.training.log.log_types import Logger

class TensorboardLogger(Logger):
  def __init__(self, path):
    self.writer = SummaryWriter(path)

  def log_image(self, name, data, step):
    self.writer.add_image(name, data, step)

  def log_image_batch(self, name, data, step):
    self.writer.add_images(name, data, step)

  def log_number(self, name, data, step):
    self.writer.add_scalar(name, data, step)

  def log_text(self, name, data, step):
    self.writer.add_text(name, data, step)

  def log_figure(self, name, data, step):
    self.writer.add_figure(name, data, step)

  def log_embedding(self, name, data, step):
    self.writer.add_embedding(name, data, step)
