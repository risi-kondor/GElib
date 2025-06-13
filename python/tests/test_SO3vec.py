import torch
import gelib as g

A=g.SO3vec.randn(1,{1:2,2:2})
Ad=g.SO3vec.randn(1,{1:2,2:2})
B=g.SO3vec.randn(3,{2:1,3:1})
print(A.__repr__())
print(A)
print(B)

R0=g.SO3element.random()
A1=A.apply(R0)
print(A1)

C=g.CGproduct(A,A)
print(C)

E=g.CGproduct(A,B)
print(E)

D=g.DiagCGproduct(A,Ad,2)
print(D)

