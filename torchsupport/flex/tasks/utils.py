import functools

from torchsupport.flex.context.context import RedirectContext

def redirect(step, *args, **kwargs):
  ctx = RedirectContext()
  # TODO

def namespace_select(f, namespace, names, knames):
  return f(
    *[namespace[name] for name in names],
    **{name: namespace[knames[name]] for name in knames}
  )

def select(f, *names, **knames):
  return functools.partial(namespace_select, f, names, knames)

def aux_chain(functions, *args, **kwargs):
  for f in functions:
    f(*args, **kwargs)

def chain(*functions):
  return functools.partial(aux_chain, functions)

def aux_with_name(ctx, name, function, *args, **kwargs):
  with ctx.switch(name):
    return function(*args, **kwargs)

def with_name(ctx, name, function):
  return functools.partial(aux_with_name, ctx, name, function)

def aux_compose(functions, **kwargs):
  tmp = []
  for f in functions:
    tmp = f(*tmp, **kwargs)

def compose(*functions):
  return functools.partial(aux_compose, functions)

def parallel_steps(ctx=None, **kwargs):
  to_chain = []
  for name, step in kwargs.items():
    to_chain.append(with_name(ctx, name, step))
  return chain(*to_chain)

def composed_steps(ctx=None, **kwargs):
  to_chain = []
  for name, step in kwargs.items():
    to_chain.append(with_name(ctx, name, step))
  return compose(*to_chain)

def noop(*args, **kwargs):
  pass
