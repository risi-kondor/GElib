import gelib as g
from e3nn import o3
import torch
import math
import time

torch.set_printoptions(sci_mode=False)
device="cuda"

def findb(c):
    if(c<32):
        return [1,c]
    return [int(c/32),32]


l1=2
nc1=10

l2=2
nc2=10

rho1=o3.Irreps(str(nc1)+"x"+str(l1)+"e")
rho2=o3.Irreps(str(nc2)+"x"+str(l2)+"e")

x1=rho1.randn(-1)
#print(x1.size())

x2=rho2.randn(-1)
#print(x2)

tp=o3.FullTensorProduct(rho1,rho2)
#print(tp)

z=tp(x1,x2)
#print(z.size())
#print(z)

for l in [2,3,5,7,9,11,13,15]:

    for nc in [1,4,16,64,256]:

        rho1=o3.Irreps(str(nc)+"x"+str(l)+"e")
        rho2=o3.Irreps(str(nc)+"x"+str(l)+"e")
        tp=o3.FullTensorProduct(rho1,rho2).to(device)

        x1=rho1.randn(-1,device=device)
        x2=rho2.randn(-1,device=device)

        [b,c]=findb(nc)
        print(b,c)
        gx1=g.SO3vec.randn(b*b,{l:c},device=device)
        gx2=g.SO3vec.randn(b*b,{l:c},device=device)
    
        niter=100

        start=time.time()
        for iter in range(niter):
            z=tp(x1,x2)
        e3nn_time=time.time()-start

        start=time.time()
        for iter in range(niter):
            gz=g.CGproduct(gx1,gx2)
        GElib_time=time.time()-start

        print("nc=",nc," l=",l," E3NN/GElib: ",(4*e3nn_time)/GElib_time)
