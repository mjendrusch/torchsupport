import inspect
import argparse
import re
import json

def get_args(method):
  """
  Helper to get the arguments (positional and keywords) 
  from the parameters of any function or class
  Args:
    method [function]: 
  returns:
    args, kwargs
  """
  sig = inspect.signature(method)
  parameters = sig.parameters
  args = []
  kwargs = []
  for key in parameters:
    param = parameters[key]
    if param.default is not param.empty:
      kwargs.append(param)
    else:
      args.append(param)
  return args, kwargs

def get_docs(method, names):
  """
  Tries to extract useful parameter descriptions from docstrings
  Expected docstring format Google (name_of_param: description)
  Args:
    method: The method to extract from
    names: an iterable containing strings for the parameter names
  returns: 
    a mapping of parameter name to docstring line
  """
  doc = inspect.getdoc(method)
  lines = [
    line.strip()
    for line in doc.split("\n")
  ]
  pattern = r"^(?::param +)?{name}\b[ ]*(?:\(.*\)|\[.*\])?:\s*(.*)$"
  docs = {}
  for name in names:
    regex = re.compile(pattern.format(name=name))
    docs[name] = ""
    for line in lines:
      match = regex.match(line)
      if match:
        docs[name] = match.group(1)
  return docs

def _maybe(data, default):
  if data is inspect.Parameter.empty:
    return default
  return data

def add_kwarg_parse(parser, kwarg, doc, namespace=None):
  """
  Logic to add a parser flag for a kwarg stored in a Parameter object
  Supports namespacing
  """
  if namespace is None:
    namespace = ""
  name = kwarg.name
  typ = _maybe(kwarg.annotation, str)
  default = _maybe(kwarg.default, None)
  try:
    parser.add_argument(f"--{name}", default=None, type=typ, help=doc)
  except argparse.ArgumentError:
    pass
  parser.add_argument(f"--{namespace}-{name}", default=None, type=typ, help=doc)
  return default

def add_class_parse(parser, the_class, namespace=None):
  """
  Expand a parser with the kwargs from a single class (or function).
  Ignores args.
  """
  if inspect.isclass(the_class):
    method = the_class.__init__
  else:
    method = the_class
  _, kwargs = get_args(method)
  names = [kwarg.name for kwarg in kwargs]
  spaced_names = [
    namespace + "_" + name # TODO: Q? Fails when no namespace provided? 
    for name in names
  ]
  docs = get_docs(method, names)
  defaults = {}
  for kwarg, doc in zip(kwargs, docs):
    default = add_kwarg_parse(parser, kwarg, doc, namespace=namespace)
    defaults[namespace + "_" + kwarg.name] = default
  return names, spaced_names, defaults

class ClassesParser():
  """
  Convenience wrapper for the standard argparse.ArgumentParser.
  Takes a list of classes (or functions) 
  and creates a namespaced argument parser (stored in self.parser)
  with flags for all keyword arguments
  """
  def __init__(self, class_dict, json_dict=None, **kwargs):
    parser, name_dict, space_dict, default_dict = self._classes_parser(class_dict, **kwargs)
    self.option_dict = None
    if json_dict is not None:
      self.option_dict = json_dict
    self.parser = parser
    self.name_dict = name_dict
    self.space_dict = space_dict
    self.default_dict = default_dict

  def _classes_parser(self, class_dict, **kwargs):
    parser = argparse.ArgumentParser(**kwargs)
    name_dict = {}
    space_dict = {}
    default_dict = {}
    for key in class_dict:
      class_names, spaced_names, defaults = add_class_parse(parser, class_dict[key], namespace=key)
      name_dict[key] = class_names
      space_dict[key] = spaced_names
      default_dict[key] = defaults
    return parser, name_dict, space_dict, default_dict
  
  def add_argument(self, *args, **kwargs):
    """Wraps argparse.ArgumentParser.add_argument"""
    self.parser.add_argument(*args, **kwargs)

  def parse_args(self, *args, **kwargs):
    """
    Parses arguments like the standard argparse.ArgumentParser.parse_args()
    Either reads from the passed command line arguments or takes a list of strings 
    Returns a special OptionWrapper object to handle namespaces
    Args:
      args: args of argparse.ArgumentParser.parse_args
      kwargs: kwargs of argparse.ArgumentParser.parse_args
    """
    options = self.parser.parse_args(*args, **kwargs)
    return OptionWrapper(
      options, self.name_dict, self.space_dict, self.default_dict,
      option_dict=self.option_dict
    )

class OptionWrapper():
  """
  Object returned by torchsupport.utils.argparse.ClassesParser
  """
  def __init__(self, options, name_dict, space_dict,
               default_dict, option_dict=None):
    if option_dict is None:
      option_dict = vars(options)
    else:
      option_vars = vars(options)
      for key in option_vars:
        if option_vars[key] is not None:
          option_dict[key] = option_vars[key]
    for name in space_dict:
      the_dict = {}
      for keyword in space_dict[name]:
        unspace = keyword[len(name) + 1:]
        if option_dict[keyword] is not None:
          the_dict[unspace] = option_dict[keyword]
        elif option_dict[unspace] is not None:
          the_dict[unspace] = option_dict[unspace]
          option_dict[keyword] = option_dict[unspace]
        else:
          the_dict[unspace] = default_dict[name][keyword]
          option_dict[keyword] = default_dict[name][keyword]
      setattr(self, name, the_dict)
    for option in option_dict:
      setattr(self, option, option_dict[option])
    self.option_dict = option_dict

  def dump_options(self, path):
    with open(path, "w") as json_file:
      json.dump(self.option_dict, json_file)

  # TODO: For greater compatibility 
  def __getitem__(self, key):
    raise NotImplementedError
  def __setitem__(self, key, value):
    raise NotImplementedError
  def __delitem__(self, key):
    raise NotImplementedError
  def __contains__(self, key):
    raise NotImplementedError
