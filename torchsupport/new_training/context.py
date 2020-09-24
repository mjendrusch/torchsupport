from copy import copy
from collections import OrderedDict

import torch

from torchsupport.data.io import DeviceMovable, Detachable, detach, to_device
from torchsupport.data.collate import Collatable, default_collate
from torchsupport.structured.chunkable import Chunkable, scatter_chunked

class BaseContext:
  __initialised = False
  def __init__(self, _init=True, **kwargs):
    self.dict = OrderedDict(**kwargs)
    self.fields = list(self.dict.keys())
    self._initialise(_init)

  def _initialise(self, init=True):
    self.__initialised = init

  def asdict(self):
    return self.dict

  def __repr__(self):
    keyvals = [
      f"{key}={self.dict[key]}"
      for key in self.dict
    ]
    keyvals = ", ".join(keyvals)
    result = f"Context({keyvals})"
    return result

  def __getattr__(self, name):
    result = ...
    if name in super().__getattribute__("fields"):
      result = super().__getattribute__("dict")[name]
    else:
      raise AttributeError
    return result

  def __setattr__(self, name, value):
    if self.__initialised:
      if name in ["fields", "dict"]:
        object.__setattr__(self, name, value)
        return
      if name not in self.fields:
        self.fields.append(name)
      self.dict[name] = value
    else:
      object.__setattr__(self, name, value)

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

class Context(
    DeviceMovable, Detachable, Chunkable,
    Collatable, BaseContext
):
  @classmethod
  def collate(cls, inputs):
    names = inputs[0].dict.keys()
    result_dict = {
      name : default_collate([
        item.dict[name]
        for item in inputs
      ])
      for name in names
    }
    return cls(**result_dict)

  def chunk(self, targets, dim=0):
    result_dicts = [{} for target in targets]
    for name, item in self.dict.items():
      results = scatter_chunked(item, targets, dim=dim)
      for idx, result in enumerate(results):
        result_dicts[idx][name] = result
    result = map(lambda x: type(self)(**x), result_dicts)
    return list(result)

  def detach(self):
    result = self.shell_copy()
    for name, item in result.dict.items():
      if isinstance(item, Context):
        result.dict[name] = item.detach()
      else:
        result.dict[name] = detach(item)
    return result

  def move_to(self, target):
    result = self.shell_copy()
    for name, item in result.dict.items():
      if isinstance(item, Context):
        result.dict[name] = item.move_to(target)
      else:
        result.dict[name] = to_device(item, target)
    return result

  def shell_copy(self):
    result = copy(self)
    result.dict = copy(self.dict)
    result.fields = copy(self.fields)
    return result

  def bind(self, **names):
    result = self.shell_copy()
    tmp = {}
    for idx, (name, value) in enumerate(names.items()):
      tmp[idx] = result.dict[name]
      result.fields[result.fields.index(name)] = idx
      del result.dict[name]
    for idx, (name, value) in enumerate(names.items()):
      if value in result.dict:
        raise RuntimeError("Invalid bind.")
      result.dict[value] = tmp[idx]
      result.fields[result.fields.index(idx)] = value
    return result

  def merge(self, name, names):
    result = self.shell_copy()
    pick, drop = result.split(*names)
    result = drop + Context(**{name: pick})
    return result

  def split(self, *names):
    pick = self.shell_copy()
    drop = self.shell_copy()
    for name in self.fields:
      if name in names:
        del drop.dict[name]
        del drop.fields[drop.fields.index(name)]
      else:
        del pick.dict[name]
        del pick.fields[pick.fields.index(name)]
    return Context(picked=pick, dropped=drop)

  def pick(self, *names):
    return self.split(*names).picked

  def drop(self, *names):
    return self.split(*names).dropped

  def view(self, name):
    return View(name, self)

  def apply(self, name, func):
    result = self.shell_copy()
    result.dict[name] = func(result.dict[name])
    return result

  def at(self, path):
    if path is None:
      return self
    result = self
    for item in path:
      result = result.dict[item]
    return result

  def set_at(self, path, data):
    if path is None:
      return False
    result = self
    for item in path[:-1]:
      result = result.dict[item]
    result.dict[path[-1]] = data
    return True

  def add(self, **names):
    result = self.shell_copy()
    for name in names:
      if name in result.fields:
        raise RuntimeError("WTF")
      result.fields.append(name)
      result.dict[name] = names[name]
    return result

  def update(self, other):
    update = copy(self.dict)
    update.update(other.dict)
    return Context(**update)

  def __add__(self, other):
    return Context(**self.dict, **other.dict)

def ctx(**kwargs):
  return Context(**kwargs)

def data_type(data, args):
  return ctx(
    data=data,
    args=args,
    loss=ctx(),
    stats=ctx()
  )

def unpack(name, context, dictionary):
  if name in dictionary:
    return dictionary[name]
  elif name in context.dict:
    return context.dict[name]
  else:
    raise KeyError(f"Key {name} not found!")

class View(Context):
  def __init__(self, name, context):
    super().__init__(_init=False)
    self.dict = context.dict
    self.fields = context.fields
    self._context = context
    self._name = name
    self._initialise()

  def apply(self, *funcs):
    out = self.dict[self._name]
    for func in funcs:
      out = func(out)
    result = self._context.shell_copy()
    result.dict[self._name] = out
    return result

  def add(self, **names):
    result = self._context.shell_copy()
    result.dict[self._name] = result.dict[self._name].add(**names)
    return result
