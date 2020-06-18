from collections import namedtuple

import torch

class AbstractDistributor:
  def commit_trajectory(self, results):
    raise NotImplementedError("Abstract.")

  def schema(self, inputs):
    return inputs

class DefaultDistributor(AbstractDistributor):
  def commit_trajectory(self, results):
    return results

class ChunkedDistributor(DefaultDistributor):
  def __init__(self, chunk_size=10):
    super().__init__()
    self.chunk_size = chunk_size

  def stack(self, chunk):
    tmp = {}
    for item in chunk:
      item = item._as_dict()
      for field in item:
        if field not in tmp:
          tmp[field] = []
        tmp[field].append(item[field].unsqueeze(0))
    for field in tmp:
      tmp[field] = torch.cat(tmp[field], dim=0)
    return type(chunk[0])(**tmp)

  def schema(self, inputs):
    chunk = [inputs] * self.chunk_size
    result = self.stack(chunk)
    return result

  def commit_trajectory(self, results):
    if len(results) < self.chunk_size:
      results += results[-1] * (self.chunk_size - len(results))

    chunked = []
    for idx, _ in enumerate(results[:self.chunk_size + 1]):
      chunk = self.stack(results[idx:idx + self.chunk_size])
      chunked.append(chunk)
    return chunked
