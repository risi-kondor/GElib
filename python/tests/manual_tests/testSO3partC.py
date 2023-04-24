import torch
import gelib as g

u=torch.randn([1,5,2],dtype=torch.cfloat)
v=g.SO3partC.from_torch(u)
print(u)
print(v)

a=g.SO3partC.randn(1,2,3)
a.requires_grad_()

b=g.CGproduct(a,a,2)
c=b.torch()
print(b)
print(c)

c.backward(c)

print(b.obj.get_grad())
print(a.obj.get_grad())

