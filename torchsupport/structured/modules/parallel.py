import torch.nn as nn

from torchsupport.structured.chunkable import scatter_chunked_kwargs

class DataParallel(nn.DataParallel):
  def scatter(self, inputs, kwargs, device_ids):
    return scatter_chunked_kwargs(inputs, kwargs, device_ids, dim=self.dim)
