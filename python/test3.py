import torch
import gelib_torchC as gelib


# ---- Fproduct -----------------------------------------------------------------------------------------------
# Given the Fourier transform of two functions u and v on SO(3), compute the Fourier transform of uv

b=2
maxl=2

# Define two random SO3 Fourier transforms 
x=gelib.SO3Fvec.randn(b,maxl)
y=gelib.SO3Fvec.randn(b,maxl)
x.parts[1].requires_grad_()

# Compute the CG-product
z=gelib.Fproduct(x,y)

print("Fproduct:")
print(z)

z.parts[2].backward(z.parts[2])
print("Backpropagated gradient:")
print(x.parts[1].grad)

print("\n\n")
