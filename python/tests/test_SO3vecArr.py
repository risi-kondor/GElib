import torch
import gelib as g

A=g.SO3vecArr.randn(1,[2,2],{1:2,2:2})
print(A)

R0=g.SO3element.random()
A1=A.apply(R0)
print(A1)

C=g.CGproduct(A,A)
print(C)

