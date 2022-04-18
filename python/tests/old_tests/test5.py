import torch
import gelib_base
#import gelib_torchD as gelib
import gelib 
import cnine 

M=torch.tensor([[0,1,0],[0,0,0],[1,0,0]],dtype=torch.float32);
mask=cnine.Rmask1(M)
print(mask)


N=3
b=1
tau=[1,1]

# Define two random SO3vec objects  
y=gelib.SO3vecArr.randn(N,b,tau)
y.parts[1].requires_grad_()
print(y)

x=y.gather(mask)
print(x)

xg=gelib.SO3vecArr.randn(N,b,tau)
print("xg=",xg)
x.parts[1].backward(xg.parts[1])
print(y.parts[1].grad)


print("-----------------")

yp=gelib.SO3partArr.randn(N,b,1,2)
print(yp)

xp=yp.gather(mask)
print(xp)
