import inspect
import types

from copy import copy
from torchsupport.new_training.context import ctx, Context

class Composition:
  def __init__(self, functions):
    self.functions = functions

  def _requires(self):
    total_requires = {}
    total_provides = {}
    for function in self.functions:
      requires = function.requires
      provides = function.provides
    # TODO

  def __call__(self, context):
    for function in self.functions:
      context = function(context)
    return context

  def __rshift__(self, other):
    if isinstance(other, Composable):
      return Composition(self.functions + [other])
    if isinstance(other, Composition):
      return Composition(self.functions + other.functions)
    else:
      raise ValueError(f"{other} is not compatible with composition.")

class Composable:
  def __init__(self, function, path=None, requires=None, provides=None):
    self.function = function
    self.requires = requires
    self.provides = provides
    self.path = path
    self.structural = False

  @property
  def func(self):
    result = self.shell_copy()
    result.structural = True
    return result

  def shell_copy(self):
    result = copy(self)
    result.requires = copy(result.requires)
    result.provides = copy(result.provides)
    result.path = copy(result.path)
    return result

  def __get__(self, owner, typ=None):
    if owner is None:
      return self
    new_function = types.MethodType(self.function, owner)
    result = self.shell_copy()
    result.requires = {
      name : result.requires[name]
      for name in result.requires
      if name != "self"
    }
    result.function = new_function
    return result

  def bind_path(self, path):
    self.path = path

  def require(self, **kwargs):
    result = copy(self)
    result.requires = copy(result.requires)
    for name in kwargs:
      result.requires[name] = kwargs[name]
    return result

  def provide(self, *args, **kwargs):
    result = copy(self)
    result.provides = copy(result.provides)
    for idx, (key, name) in enumerate(result.provides.items()):
      if idx < len(args):
        result.provides[key] = args[idx]
      elif name in kwargs:
        result.provides[key] = kwargs[result.provides[name]]
    return result

  def run_input(self, context):
    # rebind inputs for the wrapped function:
    input_feed = {
      name : context.at(self.path).dict[self.requires[name]]
      for name in self.requires
    }
    result = self.function(**input_feed)
    if result is not None:
      if not isinstance(result, tuple):
        result = (result,)
    return result

  def run_output(self, context, result):
    # rebind outputs to integrate into the surrounding context:
    output_bind = {}
    if self.provides:
      output_bind = {
        self.provides[name] : value
        for name, value in zip(self.provides, result)
      }
    modified = self.update_context(
      context.at(self.path),
      ctx(**output_bind)
    )
    result = context.shell_copy()
    if not result.set_at(self.path, modified):
      result = modified
    return result

  def update_context(self, context, other):
    return context + other

  def context_call(self, context):
    result = self.run_input(context)
    result = self.run_output(context, result)
    return result

  def arg_call(self, *args):
    context = ctx()
    result = self.function(*args)
    result_context = self.run_output(context, result)
    if len(result_context) == 1:
      return result_context[0]
    return result_context

  def call_context(self, args):
    return len(args) == 1 and isinstance(args[0], Context)

  def __call__(self, *args):
    if self.structural:
      return self.arg_call(*args)
    return self.context_call(args[0])

  def __rshift__(self, other):
    return Composition([self, other])

Run = Composable

class Update(Composable):
  def update_context(self, context, other):
    return context.update(other)

class Loss(Composable):
  def __init__(self, function, path=None, requires=None, provides=None):
    super().__init__(
      function,
      path=["context"] + (path or []),
      requires=requires,
      provides=provides
    )

  def context_call(self, context):
    context = context.shell_copy()
    if "context" not in context:
      context = ctx(loss=0.0, context=context)
    result = self.run_input(context)

    # rebind outputs to integrate into the surrounding context:
    output_bind = {
      name : value
      for name, value in zip(self.provides, result)
    }

    losses = ctx(**output_bind)
    if "losses" not in context.context.dict:
      context.context.losses = ctx()
    context.context.losses += losses
    context.loss = losses[0]

    return context

def composable(function):
  return Composable(function)

def get_returns(function):
  function_ast = inspect.ast.parse(inspect.getsource(function))
  return_stmt = function_ast.body[0].body[-1]
  provides = {}
  if isinstance(return_stmt, inspect.ast.Return):
    value = return_stmt.value
    if isinstance(value, inspect.ast.Name):
      provides[value.id] = value.id
    elif isinstance(value, inspect.ast.Tuple):
      for idx, name in enumerate(value.elts):
        if not isinstance(name, inspect.ast.Name):
          raise RuntimeError(f"Return argument {idx} is not a Name.")
        provides[name.id] = name.id
  return provides

def _nominal(function, kind=Run, path=None):
  signature = inspect.getfullargspec(function)
  args = signature.args
  requires = {
    name : name
    for name in args
  }

  returns = signature.annotations["return"]
  if isinstance(returns, (list, tuple)):
    provides = {
      name : name
      for name in returns
    }
  elif isinstance(returns, str):
    provides = {returns : returns}
  else:
    raise NotImplementedError(
      f"Return annotation {returns} is not supported yet. "
      f"Please file an issue on GitHub."
    )

  # TODO: explore metaprogramming approach for more concise definitions.
  # provides = get_returns(function)

  return kind(
    function, requires=requires, provides=provides, path=path
  )

def nominal(kind=Run, path=None):
  def helper(function):
    return _nominal(function, kind=kind, path=path)
  return helper
