import torch
import gelib 


# ---- CG-product ---------------------------------------------------------------------------------------------
# In a full CG-product each fragment in each part of x is multiplied with each fragment of each part in y 

# Define the type
adims=[2,1]
tau=[1,1]

# Define two random SO3vec objects  
x=gelib.SO3vecArr.randn(adims,tau)
y=gelib.SO3vecArr.randn(adims,tau)
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

adims=[2,1]
tau=[1,1]

R=gelib.SO3element.uniform()
x=gelib.SO3vecArr.randn(adims,tau)
y=gelib.SO3vecArr.randn(adims,tau)

z=gelib.CGproduct(x,y)
print("CG-product:")
print(z.rotate(R))

xr=x.rotate(R)
yr=y.rotate(R)

zr=gelib.CGproduct(xr,yr)
print("CG-product:")
print(zr)


print("\n\n")



