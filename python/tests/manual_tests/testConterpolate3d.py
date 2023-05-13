import torch
import gelib as g

dev='cuda'
dev='cpu'

u=g.SO3partArr.randn(1,[10,10,10],2,16,device=dev)

M=torch.randn(4,3,3,3,device=dev)
v=u.conterpolate(M)
v
t=torch.tensor(v[0,0,0,0,:,:,1:4])
print(t)


M2=torch.randn(4,3,3,3,5,device=dev)
v2=u.conterpolateB(M2)


print(v2.size())
t2=torch.tensor(v2[0,0,0,0,:,1:4])
print(t2)
