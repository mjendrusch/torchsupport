import torch
import gc
import operator

def memory_used():
    result = {}
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            if type(obj) not in result:
                result[type(obj)] = 0
            count = 1
            for elem in list(obj.data.size()):
                count *= elem
            result[type(obj.data)] += count * obj.data.element_size()
        elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
            if type(obj.data) not in result:
                result[type(obj.data)] = 0
            count = 1
            for elem in list(obj.data.size()):
                count *= elem
            result[type(obj.data)] += count * obj.data.element_size()

    return result
