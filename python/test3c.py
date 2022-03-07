import torch
import gelib_base
#import gelib_torchC as gelib
import gelib 


# ---- CG-product ---------------------------------------------------------------------------------------------
# In a full CG-product each fragment in each part of x is multiplied with each fragment of each part in y 

# Define the type 
b=2
tau=[1,1]

# Define two random SO3vec objects  
x=gelib.SO3vec.randn(b,tau)
y=gelib.SO3vec.randn(b,tau)
x.parts[1].requires_grad_()

# Compute the CG-product
z=gelib.CGproduct(x,y)

print("CG-product:")
print(z)

z.parts[2].backward(z.parts[2])
#print("dd")
print(x.parts[1].grad)

print("\n\n")


xc=x.to(device="cuda")
yc=y.to(device="cuda")

xc.parts[1].requires_grad_()
xc.parts[1].retain_grad() # why do we need this?

#Compute the CG-product
zc=gelib.CGproduct(xc,yc)

print("CG-product on GPU:")
print(zc)

zc.parts[2].backward(zc.parts[2])
#print("dd")
print(xc.parts[1].grad)

print("\n\n")


# ---- Fmodsq ----------------------------------------------------------------------------------------------------
# Given the Fourier transform of a function u on SO(3), compute the Fourier transform of the squared modulus of u

b=2
maxl=2

x=gelib.SO3vec.Frandn(b,maxl)
x.parts[1].requires_grad_()

# Compute Fmodsq
z=gelib.Fmodsq(x,maxl)

print("Fmodsq:")
print(z)

z.parts[2].backward(z.parts[2])
print("Backpropagated gradient:")
print(x.parts[1].grad)

print("\n\n")


xc=x.to(device="cuda")
xc.parts[1].requires_grad_()
xc.parts[1].retain_grad()

# Compute Fmodsq
zc=gelib.Fmodsq(xc,maxl)

print("Fmodsq on GPU:")
print(zc)

zc.parts[2].backward(zc.parts[2])
print("Backpropagated gradient:")
print(xc.parts[1].grad)

print("\n\n")


