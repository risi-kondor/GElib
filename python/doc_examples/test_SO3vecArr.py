import torch
import gelib

v=gelib.SO3vecArr.randn(1,[2,2],{0:2,1:3,2:1})
#print(v.repr())
print(v)
print(v.getb())
print(v.tau())
print(v.parts[1])

