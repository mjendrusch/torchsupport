from functools import partial

SAVABLE_EXTENSION = {}

def _set_savable(target, data_type=None):
  SAVABLE_EXTENSION[data_type] = target
  return target

class _CompareType:
  def __init__(self, data_type):
    self.data_type = data_type

  def __lt__(self, other):
    return issubclass(self.data_type. other.data_type)

def _resolve_savable(data_type):
  candidates = []
  for key in SAVABLE_EXTENSION:
    if issubclass(data_type, key):
      candidates.append(_CompareType(key))
  return SAVABLE_EXTENSION[min(candidates).data_type]

def savable_of(data_type):
  return partial(_set_savable, data_type=data_type)

class Savable:
  @staticmethod
  def wrap(data):
    if isinstance(data, Savable):
      return data
    return _resolve_savable(type(data))(data)

  def write(self, data, name):
    pass

  def read(self, data, name):
    pass

class SaveStateError(Exception):
  pass

def is_savable(x):
  return isinstance(x, Savable) or (type(x) in SAVABLE_EXTENSION)
