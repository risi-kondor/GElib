import torch
import gelib as g

tau=g.SO3bitype([[1,2,1],[2,2,3]])

u=g.SO3bivec.randn(1,tau)
print(u)

print(u.part(2,2))

#V=list((torch.randn([1,1,3],dtype=torch.cfloat),torch.randn([1,3,2],dtype=torch.cfloat)))
#v=g.SO3bivec.from_torch(V)
#print(v)


