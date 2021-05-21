import functools

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
