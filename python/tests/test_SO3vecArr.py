import torch
import gelib as g

A=g.SO3vecArr.randn(1,[2,2],{1:2,2:2})
print(A)

C=g.CGproduct(A,A)
print(C)
