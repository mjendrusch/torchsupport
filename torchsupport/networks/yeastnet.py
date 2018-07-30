import torch
import torch.nn as nn
import torch.nn.functional as func
import copy

from torchsupport.modules.compact import Conv1x1

class Parameter(object):
    pass

class Progression(object):
    def __init__(self):
        """Progression of layer parameters."""
        self.initial_state = None
        self.state = None

    def __len__(self):
        return 0

    def __iter__(self):
        self.state = copy.deepcopy(self.initial_state)
        return self

    def done(self):
        return True

    def next_state(self):
        return None

    def __next__(self):
        while not self.done():
            yield self.next_state()
        raise StopIteration

    def parameter_type(self):
        return Parameter

class BlockedNetParameter(Parameter):
    def __init__(self, inputs, outputs, outputs1x1,
                 size, stride, padding, idx):
        self.inputs = inputs
        self.outputs = outputs
        self.outputs1x1 = outputs1x1
        self.size = size
        self.stride = stride
        self.padding = padding
        self.idx = idx

class YeastNetProgression(Progression):
    def __init__(self, inputs, outputs, start, num_layers,
                 width_cap=512):
        super(YeastNetProgression, self).__init__()
        self.input = BlockedNetParameter(
            inputs, start, 0, 7, 2, 3, 0
        )
        self.output = BlockedNetParameter(
            start * 2 ** (num_layers - 1),
            start,
            0, 3, 2, 1, num_layers
        )
        self.inputs = inputs
        self.outputs = outputs
        self.start = start
        self.num_layers = num_layers
        self.width_cap = width_cap
        self.initial_state = BlockedNetParameter(
            start, start * 4, start * 2, 5, 2, 2, 0
        )

    def __len__(self):
        return self.num_layers

    def done(self):
        return self.state.idx == self.num_layers

    def next_state(self):
        next_val = BlockedNetParameter(
            self.state.outputs1x1,
            min(self.state.outputs1x1 * 4,
                self.width_cap),
            self.state.outputs1x1 * 2,
            3, 2, 1, self.state.idx + 1
        )
        result = copy.deepcopy(self.state)
        self.state = next_val
        return result

    def parameter_type(self):
        return BlockedNetParameter

class ListProgression(Progression):
    def __init__(self, **kwargs):
        self.parameters = kwargs["parameters"]
        self.input = kwargs["input"]
        self.output = kwargs["output"]
        self.initial_state = BlockedNetParameter(*self.parameters[0])
    
    def __len__(self):
        return len(self.parameters)

    def done(self):
        return self.state.idx == len(self.parameters)
        
    def next_state(self):
        next_val = BlockedNetParameter(*self.parameters[self.state.idx + 1])
        result = copy.deepcopy(self.state)
        self.state = next_val
        return result

    def parameter_type(self):
        return BlockedNetParameter

class YeastNetArch(nn.Module):
    def __init__(self, progression,
                 activation=func.leaky_relu,
                 activation_1x1=func.leaky_relu):
        """YeastNet architecture neural network."""
        super(YeastNetArch, self).__init__()
        self.num_layers = len(progression)
        self.activation = activation
        self.activation_1x1 = activation_1x1
        input_params = progression.input
        output_params = progression.output

        self.inconv = nn.Conv2d(
            input_params.inputs,
            input_params.outputs,
            input_params.size,
            input_params.stride,
            input_params.padding
        )
        self.inbn = nn.BatchNorm2d(input_params.output)

        for idx, params in enumerate(progression):
            self.__dict__[f"module{idx}"] = Conv1x1(
                params.size,
                params.stride,
                params.inputs,
                params.outputs,
                params.outputs1x1,
                activation=self.activation,
                activation_1x1=self.activation_1x1
            )
        
        self.outmodule = Conv1x1(
            output_params.size,
            output_params.stride,
            output_params.inputs,
            output_params.outputs,
            output_params.outputs1x1,
            activation=self.activation,
            activation_1x1=lambda x: x
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.inbn(self.activation(self.inconv(x)))
        for idx in range(self.num_layers):
            x = self.__dict__[f"module{idx}"](x)
        x = self.outmodule(x)
        x = self.avg(x)
        return x