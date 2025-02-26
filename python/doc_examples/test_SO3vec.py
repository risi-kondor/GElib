import torch
import gelib

v=gelib.SO3vec.randn(1,{0:2,1:3,2:1})
#print(v.repr())
print(v)
print(v.getb())
print(v.tau())
print(v.parts[1])

v=gelib.SO3vec.Frandn(1,2)
print(v)

u=gelib.SO3vec.randn(1,{0:2,1:2})
v=gelib.SO3vec.randn(1,{0:2,1:2})
w=gelib.CGproduct(u,v)
print(w)

#w=gelib.DiagCGproduct(u,v)
#print(w)




