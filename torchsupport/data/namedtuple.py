"""Pickle-safe faux namedtuple action."""

from copy import copy

from collections import OrderedDict

class CheckArgs:
  def __init__(self, name, args):
    self.name = name
    self.fields = args
    self.sfields = sorted(args)

  def __call__(self, **kwargs):
    skeys = sorted(list(kwargs.keys()))
    if skeys != self.sfields:
      raise ValueError
    result = NamedTuple(**kwargs)
    return result

class NamedTuple:
  def __init__(self, **kwargs):
    self.dict = OrderedDict(**kwargs)
    self.fields = list(self.dict.keys())

  def asdict(self):
    return self.dict

  def __repr__(self):
    keyvals = [
      f"{key}={self.dict[key]}"
      for key in self.dict
    ]
    keyvals = ", ".join(keyvals)
    result = f"NamedTuple({keyvals})"
    return result

  def __getattr__(self, name):
    result = ...
    if name in super().__getattribute__("fields"):
      result = super().__getattribute__("dict")[name]
    else:
      raise AttributeError
    return result

  def replace(self, **kwargs):
    result = copy(self)
    for key in kwargs:
      result.dict[key] = kwargs[key]
    return result

  def __getitem__(self, index):
    if index < len(self):
      return self.dict[self.fields[index]]
    else:
      raise IndexError

  def __len__(self):
    return len(self.dict)

  def __iter__(self):
    return (
      self.dict[key]
      for key in self.dict
    )

namespace = NamedTuple

def namedtuple(name, fields):
  return CheckArgs(name, fields)
