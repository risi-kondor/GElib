import torch
import gelib as g

dev='cuda'

u=g.SO3partArr.randn(1,[10,10,10],2,16,device=dev)

M=torch.randn(4,3,3,3,device=dev)
v=u.conterpolate(M)
v
t=torch.tensor(v[0,0,0,0,:,:,1:4])
print(t)
