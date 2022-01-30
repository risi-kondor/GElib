import torch
import gelib_base
import gelib_torchC as gelib


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

#z.parts[2].backward(z.parts[2])
#print("dd")
#print(x.parts[1].grad)

print("\n\n")


# ---- Fproduct -----------------------------------------------------------------------------------------------
# Given the Fourier transform of two functions u and v on SO(3), compute the Fourier transform of uv

b=2
maxl=2

# Define two random SO3 Fourier transforms 
x=gelib.SO3vec.Frandn(b,maxl)
y=gelib.SO3vec.Frandn(b,maxl)
x.parts[1].requires_grad_()

# Compute the Fproduct
z=gelib.Fproduct(x,y)

print("Fproduct:")
print(z)

z.parts[2].backward(z.parts[2])
print("Backpropagated gradient:")
print(x.parts[1].grad)

print("\n\n")



# ---- CG-product covariance test ----------------------------------------------------------------------------

b=2
tau=[1,1]
R=gelib_base.SO3element.uniform()
x=gelib.SO3vec.randn(b,tau)
y=gelib.SO3vec.randn(b,tau)

z=gelib.CGproduct(x,y)
print("CG-product:")
print(z)
print(z.rotate(R))

xr=x.rotate(R)
yr=y.rotate(R)

zr=gelib.CGproduct(xr,yr)
print("CG-product:")
print(zr)


print("\n\n")


# ---- Fproduct covariance-----------------------------------------------------------------------------------------------
# Given the Fourier transform of two functions u and v on SO(3), compute the Fourier transform of uv

b=2
maxl=2

x=gelib.SO3vec.Frandn(b,maxl)
y=gelib.SO3vec.Frandn(b,maxl)
R=gelib_base.SO3element.uniform()

z=gelib.Fproduct(x,y,2)
print("Fproduct:")
print(z.rotate(R))

xr=x.rotate(R)
yr=y.rotate(R)

zr=gelib.Fproduct(xr,yr,2)
print("Fproduct:")
print(zr)


print("\n\n")



