from torchsupport.data.collate import DataLoader

class AbstractDataDistribution:
  def sample(self, parameters):
    raise NotImplementedError("Abstract.")

class DataDistribution(AbstractDataDistribution):
  def __init__(self, dataset, batch_size=64, num_workers=8):
    self.dataset = dataset
    self.loader = DataLoader(
      dataset,
      batch_size=batch_size,
      num_workers=num_workers,
      shuffle=True,
      drop_last=True
    )
    self.data_iter = iter(self.loader)

  def sample(self, parameters):
    result = None
    try:
      result = next(self.data_iter)
    except StopIteration:
      self.data_iter = iter(self.loader)
      result = self.sample(parameters)
    return result
