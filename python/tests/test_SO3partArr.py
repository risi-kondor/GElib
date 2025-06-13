import torch
import gelib as g

P=g.SO3partArr.randn(1,[2,2],2,3)
print(P)

print(P.getb())
print(P.getl())
print(P.getn())

A=torch.randn([2,2,2,3,4])
P=g.SO3partArr.spharm(1,A)
print(P)

R=g.SO3element.random()
Pr=P.apply(R)
print(Pr)

P=g.SO3partArr.randn(1,[2,2],1,2)
Q=g.SO3partArr.randn(1,[2,2],2,2)
R=g.CGproduct(P,Q,1)
print(R)

print(P.odot(P))
