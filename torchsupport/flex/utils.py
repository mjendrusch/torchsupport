import inspect

from torchsupport.data.namedtuple import namespace

def get_kwargs(f):
  result = []
  has_kwargs = False
  sig = inspect.signature(f)
  for key, val in sig.parameters.items():
    if val.default != inspect.Parameter.empty:
      result.append(key)
    if val.kind == inspect.Parameter.VAR_KEYWORD:
      has_kwargs = True
  return result, has_kwargs

def filter_kwargs(kwargs, **targets):
  result = {}
  for name, target in targets.items():
    result[name] = {}
    target_kwargs, has_kwargs = get_kwargs(target)
    if has_kwargs:
      result[name] = kwargs
    else:
      for key in target_kwargs:
        if key in kwargs:
          result[name][key] = kwargs[key]
  return namespace(**result)
