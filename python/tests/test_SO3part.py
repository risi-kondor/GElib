import torch
import gelib as g

P=g.SO3part.randn(1,2,3)
print(P)

print(P.getb())
print(P.getl())
print(P.getn())

A=torch.randn([2,3,4])
print(A)
P=g.SO3part.spharm(1,A)
print(P)

R0=g.SO3element.random()
P1=P.apply(R0)
print(P1)

P=g.SO3part.randn(1,1,2)
Q=g.SO3part.randn(1,2,2)
R=g.CGproduct(P,Q,1)
print(R)

S=g.DiagCGproduct(P,P,2)
print(S)

Q=g.SO3part.randn(3,2,2)
R=g.CGproduct(Q,P,1)
print(R)
