import torch
import gelib_base
#import gelib_torchD as gelib
import gelib 


# ---- CG-product ---------------------------------------------------------------------------------------------
# In a full CG-product each fragment in each part of x is multiplied with each fragment of each part in y 

# Define the type
N=3
b=1
tau=[1,1]

# Define two random SO3vec objects  
x=gelib.SO3vecArr.randn(N,b,tau)
y=gelib.SO3vecArr.randn(N,b,tau)
x.parts[1].requires_grad_()

# Compute the CG-product
z=gelib.CGproduct(x,y)

print("CG-product:")
print(z)

z.parts[2].backward(z.parts[2])
print(x.parts[1].grad)

print("\n\n")


# ---- CG-product covariance test ----------------------------------------------------------------------------
print("CG-product covariance test\n")

N=3
b=2
tau=[1,1]
R=gelib_base.SO3element.uniform()
x=gelib.SO3vecArr.randn(N,b,tau)
y=gelib.SO3vecArr.randn(N,b,tau)

z=gelib.CGproduct(x,y)
print("CG-product:")
print(z.rotate(R))

xr=x.rotate(R)
yr=y.rotate(R)

zr=gelib.CGproduct(xr,yr)
print("CG-product:")
print(zr)


print("\n\n")



