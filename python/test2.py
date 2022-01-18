import torch
import gelib_torchB as gelib


# ---- Full CG-product ---------------------------------------------------------------------------------------
# In a full CG-product each fragment in each part of x is multiplied with each fragment of each part in y 

# Define the type 
tau=[1,1]

# Define two random SO3vec objects  
x=gelib.SO3vec.randn(tau)
y=gelib.SO3vec.randn(tau)
x.parts[1].requires_grad_()

# Compute the CG-product
z=gelib.FullCGproduct(x,y)

print("Full CG-product:")
print(z)

z.parts[2].backward(z.parts[2])
print(x.parts[1].grad)

print("\n\n")

# ---- Diagonal CG-product -----------------------------------------------------------------------------------
# In a diagonal CG-product each fragment in each part of x is multiplied with the corresponding fragment in y 

# Define the type 
tau=[1,1]

# Define two random SO3vec objects  
x=gelib.SO3vec.randn(tau)
y=gelib.SO3vec.randn(tau)
x.parts[1].requires_grad_()

# Compute the CG-product
z=gelib.DiagCGproduct(x,y)

print("Diagonal CG-product:")
print(z)

z.parts[2].backward(z.parts[2])
print(x.parts[1].grad)

print("\n\n")


# ---- Blockwise CG-product ---------------------------------------------------------------------------------------
# In a blockwise CG-product each fragment in each block in each part of x is multiplied with each fragment
# in each part of y in the SAME BLOCK 

#Define the maximum l
maxl=1

# Now the two vectors are define to be n channel Fourier vectors, i.e., the part with index l has n blocks
# each consisting of 2l+1 fragments 
x=gelib.SO3vec.Frandn(maxl,2)
y=gelib.SO3vec.Frandn(maxl,2)
x.parts[1].requires_grad_()

# Compute the CG-product
z=gelib.BlockwiseCGproduct(x,y)

print("Blockwise CG-product:")
print(z)

z.parts[2].backward(z.parts[2])
print(x.parts[1].grad)

print("\n\n")


# ---- Fourier conjugate -------------------------------------------------------------------------------------
# Given the Fourier transform of f, return the Fourier transform of conj(f)

x=gelib.SO3vec.Frandn(maxl,2)
x.parts[1].requires_grad_()

# Compute the Fourier conjugate
z=gelib.FourierConjugate(x)

print("Fourier conjugate:")
print(z)

z.parts[2].backward(z.parts[2])
print(x.parts[1].grad)

print("\n\n")




# exec(open("test2.py").read())





