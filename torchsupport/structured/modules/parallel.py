import torch.nn as nn

from torchsupport.structured.chunkable import scatter_chunked_kwargs
from torchsupport.data.collate import gather_collated

class DataParallel(nn.DataParallel):
  def scatter(self, inputs, kwargs, device_ids):
    return scatter_chunked_kwargs(inputs, kwargs, device_ids, dim=self.dim)

  def gather(self, outputs, output_device):
    return gather_collated(outputs, output_device, dim=self.dim)
