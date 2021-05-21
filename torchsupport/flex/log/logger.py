import warnings
from torchsupport.flex.log.log_types import LogType

class Logger:
  def log_image(self, name, data, step):
    raise NotImplementedError("Abstract")

  def log_image_batch(self, name, data, step):
    raise NotImplementedError("Abstract")

  def log_number(self, name, data, step):
    raise NotImplementedError("Abstract")

  def log_text(self, name, data, step):
    raise NotImplementedError("Abstract")

  def log_figure(self, name, data, step):
    raise NotImplementedError("Abstract")

  def log_embedding(self, name, data, step):
    raise NotImplementedError("Abstract")

  def log(self, name, data, step):
    if isinstance(data, (float, int)):
      self.log_number(name, data, step)
    elif isinstance(data, str):
      self.log_text(name, data, step)
    elif isinstance(data, LogType):
      data.log(self, name, step)
    else:
      warnings.warn(f"{name} of type {type(data)} could not be logged.\n"
                    f"Consider implementing a custom LogType.")
