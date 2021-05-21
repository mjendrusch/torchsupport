import torch

class LogType:
  @property
  def data(self):
    return 0

  def log(self, logger, name, step):
    raise NotImplementedError("Abstract")

class LogImage(LogType):
  def __init__(self, img):
    super().__init__()
    if torch.is_tensor(img):
      img = img.detach().cpu()
    if img.max() > 1.0 or img.min() < 0.0:
      img = (img - img.min()) / (img.max() - img.min())
    self.img = img

  def log(self, logger, name, step):
    if self.img.dim() > 3:
      logger.log_image_batch(name, self.img, step)
    else:
      logger.log_image(name, self.img, step)

class LogNumber(LogType):
  def __init__(self, number):
    super().__init__()
    if torch.is_tensor(number):
      number = float(number.detach().cpu())
    self.number = number

  def log(self, logger, name, step):
    logger.log_number(name, self.number, step)

class LogText(LogType):
  def __init__(self, text):
    super().__init__()
    self.text = text

  def log(self, logger, name, step):
    logger.log_text(name, self.text, step)

class LogFigure(LogType):
  def __init__(self, figure):
    super().__init__()
    self.figure = figure

  def log(self, logger, name, step):
    logger.log_figure(name, self.figure, step)

class LogEmbedding(LogType):
  def __init__(self, embedding):
    super().__init__()
    if torch.is_tensor(embedding):
      embedding = embedding.detach().cpu()
    embedding = embedding.reshape(embedding.shape[0], -1)
    self.embedding = embedding

  def log(self, logger, name, step):
    logger.log_embedding(name, self.embedding, step)
