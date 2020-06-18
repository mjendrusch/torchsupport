from collections import namedtuple

import torch
import torch.multiprocessing as mp

from torchsupport.interacting.control import ReadWriteControl

class AbstractBuffer:
  def __getitem__(self, index):
    raise NotImplementedError("Abstract.")

  def __len__(self):
    raise NotImplementedError("Abstract.")

  def append(self, items):
    raise NotImplementedError("Abstract.")

  def sample(self, size):
    raise NotImplementedError("Abstract.")

class BufferControl(ReadWriteControl):
  def __init__(self, owner):
    super().__init__(owner)
    self.item_pointer = mp.Value("l", 0)

class TensorBuffer:
  def __init__(self, *shape, size=100000, dtype=torch.float):
    self.ctrl = BufferControl(self)

    self.memory = torch.zeros(size, *shape, dtype=dtype).share_memory_()
    self.valid = torch.zeros(size, dtype=torch.bool).share_memory_()
    self.valid_index = self.valid.nonzero().view(-1)
    self.insert_pointer = 0

  @property
  def size(self):
    return self.memory.size(0)

  def data_size(self, data):
    if torch.is_tensor(data):
      dshape = data.shape
      shape = self.memory.shape
      if dshape[1:] == shape[1:]:
        return data, dshape[0]
      elif dshape == shape[1:]:
        return data.unsqueeze(0), 1
      else:
        message = f"Invalid shape: {dshape} " \
                  f"for TensorBuffer shape: {shape[1:]}"
        raise ValueError(message)
    else:
      data = torch.cat([
        self.data_size(item)[0]
        for item in data
      ], dim=0)
      return self.data_size(data)

  def invalidate(self, index):
    self.valid[index] = 0
    self.valid_index = self.valid.nonzero().view(-1)

  def raw_sample(self, indices):
    index = self.valid_index[indices]
    result = self.memory[index]
    return result

  def sample(self, size):
    with self.ctrl.read:
      valid_indices = torch.randint(self.valid_index.size(0), (size,))
      return self.raw_sample(valid_indices)

  def _update_pointer(self, count):
    self.insert_pointer = (self.insert_pointer + count) % self.size

  def _update_valid(self):
    self.valid_index = self.valid.nonzero().view(-1)

  def pull_changes(self):
    if self.ctrl.changed:
      self.insert_pointer = self.ctrl.item_pointer.value
      self._update_valid()
      self.ctrl.advance()

  def push_changes(self):
    self.ctrl.change()
    self.ctrl.item_pointer.value = self.insert_pointer

  def position_for_append(self, data):
    data, size = self.data_size(data)
    item_position = self.insert_pointer
    reach = item_position + size
    if reach > self.size:
      item_position = 0
    nuke_next_n = size
    return item_position, nuke_next_n

  def prepare_for_append(self, item_position, nuke_next_n):
    self.insert_pointer = item_position
    self.valid[item_position:item_position + nuke_next_n] = 0

  def raw_append(self, tensor):
    data, size = self.data_size(tensor)
    self.memory[self.insert_pointer:self.insert_pointer + size] = tensor
    self.valid[self.insert_pointer:self.insert_pointer + size] = 1
    self._update_pointer(size)
    self._update_valid()

  def _append_item_tensor(self, item):
    item = item.unsqueeze(0)
    item_position, nuke_next_n = self.position_for_append(item)
    self.prepare_for_append(item_position, nuke_next_n)
    self.raw_append(item)

  def _append_batch_tensor(self, batch):
    batch_size = batch.size(0)
    circular_slice = (self.insert_pointer + torch.arange(batch_size)) % self.size
    self.memory[circular_slice] = batch
    self.valid[circular_slice] = 1
    self._update_pointer(batch_size)

  def _append_generic_iterable(self, iterable):
    for item in iterable:
      self._append_dispatch(item)

  def _append_dispatch(self, item):
    data, _ = self.data_size(item)
    self._append_batch_tensor(data)

  def append(self, item):
    with self.ctrl.write:
      self._append_dispatch(item)

  def __getitem__(self, index):
    with self.ctrl.read:
      return self.memory[index]

  def __len__(self):
    with self.ctrl.read:
      return len(self.valid_index)

class TensorSequenceBuffer(TensorBuffer):
  INVALID_SEQUENCE = -1
  def __init__(self, *shape, size=100000, dtype=torch.float):
    super().__init__(*shape, size=size, dtype=dtype)
    self.sequence_start = self.INVALID_SEQUENCE * torch.ones(size, dtype=torch.long)
    self.sequence_end = self.INVALID_SEQUENCE * torch.ones(size, dtype=torch.long)
    self.sequence_pointer = 0
    self.valid_sequence = self.sequence_start != self.INVALID_SEQUENCE
    self.valid_sequence_index = self.valid_sequence.nonzero()

  def _fits_end(self, size):
    return self.insert_pointer + size < self.size

  def raw_append(self, tensor):
    size = tensor.size(0)
    self.sequence_start[self.sequence_pointer] = start = self.insert_pointer
    self.sequence_end[self.sequence_pointer] = end = self.insert_pointer + size
    self._update_pointer(size)
    self.sequence_pointer = self.sequence_pointer + 1
    self.memory[start:end] = tensor
    self._update_valid_sequence()

  def _append_sequence_tensor(self, sequence):
    self._accomodate_overlap(sequence)
    self.raw_append(sequence)

  def _append_sequence_iterable(self, iterable):
    for item in iterable:
      if item.shape[1:] == self.memory.shape[1:]:
        self._append_sequence_tensor(item)
      elif item.shape == self.memory.shape[1:]:
        item = item.unsqueeze(0)
        self._append_sequence_tensor(item)
      else:
        message = f"Invalid shape: {item.shape} " \
                  f"for TensorBuffer shape: {self.memory.shape[1:]}"
        raise ValueError(message)

  def _append_dispatch(self, data):
    if torch.is_tensor(data):
      self._append_sequence_tensor(data)
    else:
      self._append_sequence_iterable(data)

  def _update_valid_sequence(self):
    self.valid_sequence = self.sequence_start != self.INVALID_SEQUENCE
    self.valid_sequence_index = self.valid_sequence.nonzero()

  def _accomodate_overlap(self, data):
    item_position, nuke_next_n = self.position_for_append(data)
    self.prepare_for_append(item_position, nuke_next_n)

  def _find_first_writable(self, item_position):
    result = self.sequence_start[item_position]
    if result is None:
      if item_position == 0:
        result = 0
      else:
        result = self.sequence_end[item_position - 1]
    return result

  def position_for_append(self, data):
    size = data.size(0)
    item_position = self.sequence_pointer
    if not self._fits_end(size):
      item_position = 0

    first_writable_position = self._find_first_writable(item_position)
    reach = first_writable_position + size
    nuke_next_n = (self.sequence_start[item_position:] < reach).sum()

    return item_position, nuke_next_n

  def prepare_for_append(self, item_position, nuke_next_n):
    self.sequence_pointer = item_position
    self.insert_pointer = self._find_first_writable(item_position)
    nuke_start = self.sequence_pointer
    nuke_end = self.sequence_pointer + nuke_next_n
    self.sequence_start[nuke_start:nuke_end] = self.INVALID_SEQUENCE
    self.sequence_end[nuke_start:nuke_end] = self.INVALID_SEQUENCE

  def raw_sample(self, indices):
    result = []
    for item in indices:
      result.append(self.__getitem__(item))
    return result

  def sample(self, size):
    indices = torch.randint(len(self), (size,))
    return self.raw_sample(indices)

  def __getitem__(self, index):
    start = self.sequence_start[index]
    end = self.sequence_end[index]
    return self.memory[start:end]

  def __len__(self):
    return len(self.valid_sequence_index)

class NoneBuffer(AbstractBuffer):

  def raw_sample(self, indices):
    return None

  def sample(self, size):
    return None

  def pull_changes(self):
    pass

  def push_changes(self):
    pass

  def raw_append(self, data):
    pass

  def data_size(self, data):
    return None, None

  def position_for_append(self, data):
    return None, None

  def prepare_append(self, position, nuke_next_n):
    pass

  def append(self, data):
    pass

  def __getitem__(self, index):
    return None

  def __len__(self):
    return 1000000

class CombinedBuffer(AbstractBuffer):
  def __init__(self, **buffers):
    self.ctrl = BufferControl(self)
    self.buffers = buffers
    self.data_type = namedtuple("Data", list(buffers.keys()))

  def raw_sample(self, indices):
    result = self.data_type(**{
      key: self.buffers[key].raw_sample(indices)
      for key in self.buffers
    })
    return result

  def sample(self, size):
    with self.ctrl.read:
      indices = torch.randint(len(self), (size,))
      return self.raw_sample(indices)

  def pull_changes(self):
    for key in self.buffers:
      self.buffers[key].pull_changes()

  def push_changes(self):
    for key in self.buffers:
      self.buffers[key].push_changes()

  def raw_append(self, data):
    for key in self.buffers:
      buffer = self.buffers[key]
      buffer.raw_append(data[key])

  def data_size(self, data):
    items = {}
    if isinstance(data, tuple) and hasattr(data, "_asdict"):
      data = data._asdict()
    result_size = None
    for key in self.buffers:
      buffer = self.buffers[key]
      if isinstance(buffer, NoneBuffer):
        items[key] = None
        continue
      item, size = buffer.data_size(data[key])
      if result_size is None:
        result_size = size
      elif size != result_size:
        message = f"Inconsistent size for append. " \
                  f"Expected: {result_size} but got: {size}."
        raise ValueError(message)
      items[key] = item
    return items, result_size

  def position_for_append(self, data):
    positions = []
    nukes = []

    for key in self.buffers:
      buffer = self.buffers[key]
      if isinstance(buffer, NoneBuffer):
        continue
      p, n = buffer.position_for_append(data[key])
      positions.append(p)
      nukes.append(n)
    position = min(positions)
    nuke_next_n = max(nukes)
    return position, nuke_next_n

  def prepare_append(self, position, nuke_next_n):
    for key in self.buffers:
      buffer = self.buffers[key]
      if isinstance(buffer, NoneBuffer):
        continue
      buffer.prepare_for_append(position, nuke_next_n)

  def append(self, data):
    with self.ctrl.write:
      data, _ = self.data_size(data)
      position, nuke_next_n = self.position_for_append(data)
      self.prepare_append(position, nuke_next_n)
      self.raw_append(data)

  def __getitem__(self, index):
    with self.ctrl.read:
      result = self.data_type(**{
        key: self.buffers[key][index]
        for key in self.buffers
      })
      return result

  def __len__(self):
    return min(map(len, [self.buffers[key] for key in self.buffers]))

def SchemaBuffer(schema, size):
  result = ...
  if schema is None:
    result = NoneBuffer()
  elif torch.is_tensor(schema):
    shape = schema.shape
    dtype = schema.dtype
    result = TensorBuffer(*shape, size=size, dtype=dtype)
  elif isinstance(schema, dict):
    result = CombinedBuffer(**{
      key : SchemaBuffer(schema[key], size)
      for key in schema
    })
  elif isinstance(schema, tuple) and hasattr(schema, "_asdict"):
    result = SchemaBuffer(schema._asdict(), size)
  else:
    raise ValueError(f"{type(schema)} is not a valid schema type.")
  return result
