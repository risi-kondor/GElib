import torch
import torch.nn as nn

import sys
sys.path.insert(1, '../models')

import gelib_base
import gelib_torchC as gelib
from gelib_torchC import *

# Random seed
torch.manual_seed(123456789)

# Create model
class Test_Model(torch.nn.Module):
    def __init__(self):
        super(Test_Model, self).__init__()

    def forward(self, inputs):
        outputs = gelib.CGproduct(inputs, inputs)
        return outputs

model = Test_Model()

# Run the model
batch_size = 10
maxl = 15
input_tau = [1 for l in range(maxl + 1)]
inputs = gelib.SO3vec.randn(batch_size, input_tau)
outputs = model(inputs)

# Rotate the inputs and run the model again
R = gelib_base.SO3element.uniform()
inputs_rot = inputs.rotate(R)
outputs_rot = model(inputs_rot)

# Check the difference
output_str = []

def search_value(tensor, value):
    tensor = tensor.detach().cpu().numpy()
    assert len(tensor.shape) == 3
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            for v in range(tensor.shape[2]):
                if tensor[i, j, v] == value:
                    return [i, j, v]
    return None

assert(len(outputs.parts) == len(outputs_rot.parts))
for l in range(len(outputs.parts)):
    first = torch.view_as_complex(outputs.rotate(R).parts[l])
    second = torch.view_as_complex(outputs_rot.parts[l])

    # print(first.is_contiguous())
    # print(second.is_contiguous())
    
    # print(first.size())
    # print(second.size())

    diff = torch.norm(first - second, p = 1) / second.numel()

    '''
    print(l, '---------------------')
    print('Tensor size:', first.size())
    print('Min, max of the first tensor:', torch.min(torch.abs(first)).item(), ' ', torch.max(torch.abs(first)).item())
    print('Min, max of the second tensor:', torch.min(torch.abs(second)).item(), ' ', torch.max(torch.abs(second)).item())
    print('MAE:', diff)
    '''

    output_str.append(str(l) + ' ---------------------')
    output_str.append('Tensor size: ' + str(first.size()))
    output_str.append('Min, max of the first tensor: ' + str(torch.min(torch.abs(first)).item()) + ' ' + str(torch.max(torch.abs(first)).item()))
    output_str.append('Position of the max value of the first tensor:' + str(search_value(torch.abs(first), torch.max(torch.abs(first)).item())))
    output_str.append('Min, max of the second tensor: ' + str(torch.min(torch.abs(second)).item()) + ' ' + str(torch.max(torch.abs(second)).item()))
    output_str.append('Position of the max value of the second tensor:' + str(search_value(torch.abs(second), torch.max(torch.abs(second)).item())))
    output_str.append('MAE: ' + str(diff))

print('Summary -------------------------------------------')
for s in output_str:
    print(s)

print('Done')
