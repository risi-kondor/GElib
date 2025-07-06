import torch
import gelib as g

P=g.O3part.randn(1,(2,-1),3)
print(P)

print(P.getb())
print(P.getl())
print(P.getp())
print(P.getn())

A=torch.randn([2,3,4])
P=g.O3part.spharm((1,-1),A)
print(P)

R0=g.O3element.random()
P1=P.apply(R0)
print(P1)

print("CGproduct:")
P=g.O3part.randn(1,(1,-1),2)
Q=g.O3part.randn(1,(2,1),2)
R=g.CGproduct(P,Q,(1,-1))
print(R)

print("DiagCGproduct:")
S=g.DiagCGproduct(P,P,(2,1))
print(S)

print("CGproduct:")
Q=g.O3part.randn(3,(2,1),2)
R=g.CGproduct(Q,P,(1,-1))
print(R)
