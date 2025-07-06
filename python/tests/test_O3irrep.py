import torch
import gelib as g

R0=g.O3element.random()
R1=g.O3element.random()
R2=R0*R1

rho=g.O3irrep(2,1)

A=rho.matrix(R0)
B=rho.matrix(R1)
C=rho.matrix(R2)

print(torch.matmul(A,B))
print(C)
