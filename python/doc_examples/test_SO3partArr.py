import torch
import gelib


P=gelib.SO3partArr.randn(1,[2,2],2,3)
print(P)

print(P.getb())
print(P.get_adims())
print(P.getl())
print(P.getn())

B=gelib.CGproduct(P,P,2)
print(B)

import cnine
C=torch.tensor([[0,1,0],[0,0,0],[1,0,0]],dtype=torch.float32);
mask=cnine.Rmask1(C)
