import torch

def flatten(input, batch=False):
    if batch:
        return input.view(input.size()[0], -1)
    else:
        return input.view(-1)

def batchexpand(input, batch):
    return input.unsqueeze(0).expand(
        batch.size()[0],
        *input.size()
    )
