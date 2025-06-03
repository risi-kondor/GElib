import numpy
import torch
import cnine

a=cnine.cscalar()
print(a.str())

dims=cnine.gdims([4,4])
print(dims.str())

#T=cnine.ctensor(dims,cnine.fill_zero())
T=cnine.ctensor.sequential(dims,-1,0)
print(T)
print(T.get_k())
print(T.get_dims())
print(T.get_dim(0))
print(T.get(1,2))

T.set_value(1,2,99)
print(T)

print(T.transp())

M=torch.tensor([[0,1,0],[0,0,0],[1,1,1]],dtype=torch.float32);
mask=cnine.Rmask1(M)
print(mask)

#print(T[1,2])

