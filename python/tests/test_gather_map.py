import torch
import gelib_base as gb
import gelib as g

m=gb.gather_map.random(5,5)
print(m)

M=torch.tensor([[4,3,9,1],[5,5,2,7]],dtype=torch.int32)
m=gb.gather_map.from_matrix(M.t(),10,10)
print(m)


      
