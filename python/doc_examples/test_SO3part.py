import torch
import gelib


P=gelib.SO3part.randn(1,2,3)
print(P)

print(P.getb())
print(P.getl())
print(P.getn())
print(P.__repr__())

B=torch.randn([1,3,2],dtype=torch.complex64)
print(type(B))
P=gelib.SO3part(B)
print(P)

A=torch.randn([2,3,4])
P=gelib.SO3part.spharm(1,A)
print(P)

P=gelib.SO3part.randn(1,1,2)
Q=gelib.SO3part.randn(1,2,2)
R=gelib.CGproduct(P,Q,1)
print(R)

R=gelib.DiagCGproduct(P,Q,1)
print(R)

   
