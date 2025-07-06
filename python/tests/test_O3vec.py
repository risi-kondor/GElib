import torch
import gelib as g

A=g.O3vec.randn(1,{(1,1):2,(2,-1):2})
Ad=g.O3vec.randn(1,{(1,1):2,(2,-1):2})
B=g.O3vec.randn(3,{(2,1):1,(3,1):1})
print(A.__repr__())
print(A)
print(B)

print("Rotate:")
R0=g.O3element.random()
A1=A.apply(R0)
print(A1)

print("CGproduct:")
C=g.CGproduct(A,A)
print(C)

print("CGproduct:")
E=g.CGproduct(A,B)
print(E)

print("DiagCGproduct:")
D=g.DiagCGproduct(A,Ad,2)
print(D)

