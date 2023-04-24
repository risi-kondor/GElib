import torch
import gelib as g

u=g.SO3vecC.randn(1,[2,2,2])
print(u)

V=list((torch.randn([1,1,3],dtype=torch.cfloat),torch.randn([1,3,2],dtype=torch.cfloat)))
v=g.SO3vecC.from_torch(V)
print(v)


