import torch
import torch.nn as nn
import torch.nn.functional as func

class DyNet(nn.Module):
    r'''Implements dynamic convolution as introduced by Zhang et al. 
        (https://arxiv.org/abs/2004.10694). This version is implemented with conv2D.
    Args: 
        in_size (int): input size
        out_size (int): desired output size
        k_number (int): number of kernels 
        k_size (int): size of the kernel
        **kwargs: for details see Conv2d
    '''
    def __init__(self, in_size, out_size, k_number, k_size, **kwargs):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.k_number = k_number
        self.k_size = k_size
        self.kwargs = kwargs
        self.weight = nn.Parameter(torch.randn(k_number, out_size, in_size, k_size, k_size))
        self.fc = nn.Linear(in_size, k_number)

    def forward(self, inputs):   
        avg = func.adaptive_avg_pool2d(inputs, 1) # (batch_size, in_size, 1, 1)
        avg = avg.view(inputs.size(0), -1)  
        kernel_weights = self.fc(avg)
        kernel_weights = kernel_weights[:, :, None, None, None, None]
        dyn_kernels = torch.sum(kernel_weights * self.weight[None], dim=1)   
        dyn_kernels = dyn_kernels.view(*dyn_kernels.shape[1:])

        batch_size = inputs.size(0)
        inputs = inputs.view(1, -1, *inputs.shape[2:])
        result = func.conv2d(inputs, groups=batch_size, weight=dyn_kernels, **self.kwargs)
        return result.view(batch_size, -1, *result.shape[2:])
        