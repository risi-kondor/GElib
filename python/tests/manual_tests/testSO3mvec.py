import torch
import gelib 


# ---- CG-product ---------------------------------------------------------------------------------------------
# In a full CG-product each fragment in each part of x is multiplied with each fragment of each part in y 

# Define the type 
b=1
k=2
tau=[1,1]

# Define two random SO3vec objects  
x=gelib.SO3mvec.randn(b,k,tau)
y=gelib.SO3mvec.randn(b,k,tau)
x.parts[1].requires_grad_()

# Compute the CG-product
z=gelib.CGproduct(x,y)

print(x)
print("CG-product:")
print(z)

z.parts[2].backward(z.parts[2])
print(x.parts[1].grad)

print("\n\n")


# ---- Fproduct -----------------------------------------------------------------------------------------------
# Given the Fourier transform of two functions u and v on SO(3), compute the Fourier transform of uv

b=2
maxl=2

# Define two random SO3 Fourier transforms 
x=gelib.SO3mvec.Frandn(b,k,maxl)
y=gelib.SO3mvec.Frandn(b,k,maxl)
x.parts[1].requires_grad_()

# Compute the Fproduct
z=gelib.Fproduct(x,y)

print("Fproduct:")
print(z)

z.parts[2].backward(z.parts[2])
print("Backpropagated gradient:")
print(x.parts[1].grad)

print("\n\n")


# ---- Fmodsq ----------------------------------------------------------------------------------------------------
# Given the Fourier transform of a function u on SO(3), compute the Fourier transform of the squared modulus of u

b=2
maxl=2

x=gelib.SO3mvec.Frandn(b,k,maxl)
x.parts[1].requires_grad_()

# Compute Fmodsq
z=gelib.Fmodsq(x)

print("Fmodsq:")
print(z)

z.parts[2].backward(z.parts[2])
print("Backpropagated gradient:")
print(x.parts[1].grad)

print("\n\n")


# ---- CG-product covariance test ----------------------------------------------------------------------------
print("CG-product covariance test\n")

b=2
tau=[1,1]
R=gelib.SO3element.uniform()
x=gelib.SO3mvec.randn(b,k,tau)
y=gelib.SO3mvec.randn(b,k,tau)

z=gelib.CGproduct(x,y)
print("CG-product:")
print(z.rotate(R))
print(999)

xr=x.rotate(R)
yr=y.rotate(R)

zr=gelib.CGproduct(xr,yr)
print("CG-product:")
print(zr)


print("\n\n")


# ---- Fproduct covariance test ------------------------------------------------------------------------------
print("Fproduct covariance test\n")

b=2
maxl=2

x=gelib.SO3mvec.Frandn(b,k,maxl)
y=gelib.SO3mvec.Frandn(b,k,maxl)
R=gelib.SO3element.uniform()

z=gelib.Fproduct(x,y,2)
print("Fproduct:")
print(z.rotate(R))

xr=x.rotate(R)
yr=y.rotate(R)

zr=gelib.Fproduct(xr,yr,2)
print("Fproduct:")
print(zr)


print("\n\n")


# ---- Fmodsq covariance test ------------------------------------------------------------------------------
print("Fmodsq covariance test\n")

b=2
maxl=2

x=gelib.SO3mvec.Frandn(b,k,maxl)
R=gelib.SO3element.uniform()

z=gelib.Fmodsq(x,2)
print("Fmodsq:")
print(z.rotate(R))

xr=x.rotate(R)

zr=gelib.Fmodsq(xr,2)
print("Fmodsq:")
print(zr)


print("\n\n")
for x, y in zip(zr.parts, z.rotate(R).parts):
    print(torch.allclose(x, y))



