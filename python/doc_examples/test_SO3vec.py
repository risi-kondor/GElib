import torch
import gelib

v=gelib.SO3vec.randn(1,{0:2,1:3,2:1})
print(v)
print(v.getb())
print(v.tau())
print(v.parts[1])
