import torch
import gelib as g
import numpy as np
from scipy.special import sph_harm

n=3
l=4
M=torch.randn(1,3,n)
print(M)

A=torch.zeros(2*l+1,n,dtype=torch.cfloat)
for i in range(n):
    nm=torch.norm(M[0,:,i]).item()
    x=M[0,0,i].item()/nm
    y=M[0,1,i].item()/nm
    z=M[0,2,i].item()/nm
    print(z)
    theta = np.arccos(z)  
    phi = np.arctan2(y, x)
    for m in range(-l,l+1):
        A[m+l,i]=sph_harm(m, l, phi, theta)
print(A)

B=g.SO3part.spharm(l,M)
print(B)

print(A/B)
