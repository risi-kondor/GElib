import torch
import gelib 

L=2
a=gelib.SO3vec.Frandn(1,L)
d=torch.randn([1,1,1])
d.requires_grad_()
a.parts[0]+=d
print(a)

b=gelib.CGproduct(a,a)
c=b.parts[0]
c.backward(c)

