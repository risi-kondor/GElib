import torch
import gelib as g

u=g.SO3vecArrC.randn(1,[2,2],[2,3])
print(u)

print("Part l=1:")
print(u[1])

print("Cell (0,1):")
print(u.cell([0,0,1]))

V=list((torch.randn([1,2,2,1,3],dtype=torch.cfloat),torch.randn([1,2,2,3,2],dtype=torch.cfloat)))
v=g.SO3vecArrC.from_torch(V)
print(v)


