import torch
import time

import sys
sys.path.insert(1, '../models/')
#import gelib_torchC as gelib
import gelib

torch.manual_seed(123)

# +-----+
# | CPU |
# +-----+

# Define the type 
batch = 100
maxl = 10

# Define a random SO3vec object
x = gelib.SO3vec.Frandn(batch, maxl)

for l in range(maxl + 1):
    # x.parts[l].requires_grad_()

    x.parts[l].requires_grad_().retain_grad()

# Compute the Fmodsq
start = time.time()
z = gelib.Fmodsq(x)
finish = time.time()
cpu_forward_time = (finish - start)

# Sum all elements
# out=torch.sum(z.parts[0])+torch.sum(z.parts[1])+torch.sum(z.parts[2])

# Take the norm
out = torch.norm(z.parts[0]) + torch.norm(z.parts[1]) + torch.norm(z.parts[2])

target = torch.ones(out.size())
loss = torch.nn.functional.mse_loss(out,target)

# Backward
start = time.time()
loss.backward()
finish = time.time()
cpu_backward_time = (finish - start)

# +------+
# | CUDA |
# +------+

# Should not use this initialization
# x_cuda = x.to(device = 'cuda')

x_cuda = gelib.SO3vec.Frandn(batch, maxl)

for l in range(maxl + 1):
    x_cuda.parts[l] = x.parts[l].detach().to(device = 'cuda')

    x_cuda.parts[l].requires_grad_()
    
    # x_cuda.parts[l].retain_grad()

    print(x_cuda.parts[l].grad) # Should be None

start = time.time()
z_cuda = gelib.Fmodsq(x_cuda)
finish = time.time()
cuda_forward_time = (finish - start)

out_cuda = torch.norm(z_cuda.parts[0]) + torch.norm(z_cuda.parts[1]) + torch.norm(z_cuda.parts[2])
target_cuda = torch.ones(out_cuda.size()).to(device = 'cuda')
loss_cuda = torch.nn.functional.mse_loss(out_cuda, target_cuda)

start = time.time()
loss_cuda.backward()
finish = time.time()
cuda_backward_time = (finish - start)

# +----------------+
# | Check accuracy |
# +----------------+

print('Difference in the loss (should be 0):')
print(torch.norm(loss.detach() - loss_cuda.cpu().detach()))

print('Difference in the output (should be 0):')
print(torch.norm(out.detach() - out_cuda.cpu().detach()))

print('Difference in each individual tensor:')
for l in range(maxl + 1):
    print(torch.norm(z.parts[l].detach() - z_cuda.parts[l].cpu().detach()))

print('Difference in each individual tensor gradient:')
for l in range(maxl + 1):
    # print(x_cuda.parts[l].grad) # NoneType
    # print(x.parts[l].grad)
    print(torch.norm(x.parts[l].grad.detach() - x_cuda.parts[l].grad.cpu().detach()))

print('---- Time ----')
print('CPU forward time:', cpu_forward_time)
print('CUDA forward time:', cuda_forward_time)
print('CPU backward time:', cpu_backward_time)
print('CUDA backward time:', cuda_backward_time)

print('Done')
