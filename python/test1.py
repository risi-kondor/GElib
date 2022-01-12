import torch
import gelib_torchB as gelib

L=2
a=gelib.SO3vec.Frandn(L,1)
d=torch.randn([L,1,1])
d.requires_grad_()
a.parts[0]+=d
print(a)

b=gelib.FullCGproduct(a,a)
c=b.parts[0]
c.backward(c)

