import torch
import torch.nn as nn

class Storage:
  def __init__(self):
    self.dict = {}
    self.fields = []

  def asdict(self):
    return self.dict

  def __repr__(self):
    keyvals = [
      f"{key}={self.dict[key]}"
      for key in self.dict
    ]
    keyvals = ", ".join(keyvals)
    result = f"Storage({keyvals})"
    return result

  def __getattr__(self, name):
    result = ...
    if name in super().__getattribute__("fields"):
      result = super().__getattribute__("dict")[name]
    else:
      raise AttributeError
    return result

  def prepare(self, value):
    if torch.is_tensor(value):
      return value.cpu().detach()
    return value

  def store(self, **kwargs):
    for key, val in kwargs.items():
      self.dict[key] = self.prepare(val)
    self.fields = list(self.dict.keys())

  def items(self):
    for item in self.dict.items():
      yield item

  def __len__(self):
    return len(self.dict)

  def __iter__(self):
    return (
      self.dict[key]
      for key in self.dict
    )
