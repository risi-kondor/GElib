import gelib as g
from e3nn import o3
import torch
import math
import time

torch.set_printoptions(sci_mode=False)
device="cuda"

A=torch.randn(10,3)

l=3

converter=torch.zeros([2*l+1,2*l+1],dtype=torch.cfloat)
converter[l,l]=1
afact=1.0/math.sqrt(2.0)+0j
bfact=complex(0,1.0/math.sqrt(2.0))
for m in range(1,l+1):
    converter[l+m,l-m]=afact
    converter[l+m,l+m]=afact*(1-2*(m%2))
    converter[l-m,l-m]=bfact
    converter[l-m,l+m]=-bfact*(1-2*(m%2))
#print(converter)

permuter=torch.zeros(3,3)
permuter[0,1]=1
permuter[1,2]=1
permuter[2,0]=1

Y=o3.spherical_harmonics(l,A@(permuter.t()),True)
#print(Y)

gY=torch.Tensor(g.SO3part.spharm(l,A.t().unsqueeze(0))).squeeze(0)
gYY=torch.Tensor((torch.mm(converter,gY)).real).t()

#print(gYY)



for nc in [1,4,16,64,256,1024,2048]:
    A=torch.randn(nc,3,device=device)
    gA=A.t().unsqueeze(0)
    
    for l in [2,3,5,7,9,11]:
        niter=100

        start=time.time()
        for iter in range(niter):
            Y=o3.spherical_harmonics(l,A,True)
        e3nn_time=time.time()-start

        start=time.time()
        for iter in range(niter):
            gY=g.SO3part.spharm(l,gA)
        GElib_time=time.time()-start

        print("nc=",nc," l=",l," E3NN/GElib: ",e3nn_time/GElib_time)
