import torch
import gelib as g

u=torch.randn([1,2,2,5,2],dtype=torch.cfloat)
v=g.SO3partArrC.from_torch(u)
print(u)
print(v)

a=g.SO3partArrC.randn(1,[2,2],2,3)
a.requires_grad_()

print("Cell (0,1):")
print(a.cell([0,0,1])) # correct this 

b=g.CGproduct(a,a,2)
c=b.torch()
print(b)
print(c)

c.backward(c)

print(b.obj.get_grad())
print(a.obj.get_grad())

