import torch
import gelib

u=gelib.SO3partB.randn(1,2,2)
v=gelib.SO3partB.randn(1,2,2)

print(u.__repr__())

w=gelib.CGproduct(u,v,2)

print(w)

