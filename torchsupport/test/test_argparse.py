import pytest
import argparse as py_ap
import inspect
import types
from typing import Union
from ..utils import argparse as ap

# region Examples
def example_empty():
  pass

def example_args(a, b: Union[int, float], c: int):
  pass

def example_kwargs(a=None, b=1, c='test'):
  """
  Try me
  Args:
    a: value 
    b[int]: test
    c : me
  returns:
    a sense of nothing
  """
  pass

def example_combi(pos, a=None, b=1, c='test'):
  """
  I'm a harder case:
  a slice is indicated by :
  Args:
    pos:
    a: value
    b: test
  returns:
  a list separated by colon (:)
  """
  pass

class ExampleClassEmpty():
  def __init__(self):
    pass
  def method_empty(self):
    pass  
  def method_args(self, a, b, c):
    pass
  def method_kwargs(self, a=None, b=1, c='test'):
    pass
  def method_combi(self, pos, a=None, b=1, c='test'):
    pass

class ExampleClassArgs():
  def __init__(self, a, b, c):
    pass

class ExampleClassKwargs():
  def __init__(self, a=None, b=1, c='test'):
    """
    Regular docstring Google style
    Args:
      a: some value
      b: some int
      c: default=test
    """
    pass

class ExampleClassCombi():
  def __init__(self, pos, a=None, b=1, c='test'):
    """
    Regular docstring but in sphinx style
    :param pos: I'm a regular positional argument
    :param a: some value
    :param b: some int
    :param c: default=test
    Constructors aren't known for their return value
    """
    pass

example_class_obj = ExampleClassEmpty()
# endregion

# region argparse.get_args
def test_getargs_basic():
  result = ap.get_args(example_combi)
  assert len(result) == 2

@pytest.mark.parametrize('in_func, n_args, n_kwargs', [
    (example_empty, 0, 0),
    (example_args, 3, 0),
    (example_kwargs, 0, 3),
    (example_combi, 1, 3),
    (ExampleClassEmpty, 0, 0),
    (ExampleClassArgs, 3, 0),
    (ExampleClassKwargs, 0, 3),
    (ExampleClassCombi, 1, 3),
    (example_class_obj.method_empty, 0, 0),
    (example_class_obj.method_args, 3, 0),
    (example_class_obj.method_kwargs, 0, 3),
    (example_class_obj.method_combi, 1, 3),
])
def test_getargs(in_func, n_args, n_kwargs):
  args, kwargs = ap.get_args(in_func)
  assert len(args) == n_args
  assert len(kwargs) == n_kwargs
  for arg in args:
    assert isinstance(arg, inspect.Parameter)
  for arg in kwargs:
    assert isinstance(arg, inspect.Parameter)

def test_getargs_badtypes():
  # TODO: Make the raises test more stringent
  for ugly in [0, None, 'fuck']:
    with pytest.raises(TypeError):
        ap.get_args(ugly)

# endregion

# region argparse.get_docs
def test_getdocs_simple():
  # TODO: Improve the coverage of weirder cases
  # Maybe require a prettier return
  assert ap.get_docs(example_kwargs, [*'abc']) == {'a': 'value', 'b': 'test', 'c': 'me'}

def test_getdocs_harder():
  desired = {'pos': '', 'a': 'value', 'b': 'test', 'c': ''} 
  result = ap.get_docs(example_combi, ['pos', *'abc'])
  assert result == desired

# endregion

# region argparse.ClassesParser

# endregion