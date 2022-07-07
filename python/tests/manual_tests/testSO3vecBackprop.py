import torch
import gelib 


b=1
tau=[1,1]

# Define two random SO3vec objects  
x=gelib.SO3vec.randn(b,tau)
y=gelib.SO3vec.randn(b,tau)
x.requires_grad_()

# Compute the CG-product
z=gelib.CGproduct(x,y)

print("CG-product:")
print(z)

test_vec=gelib.SO3vec.randn_like(z)
loss=z.odot(test_vec)
loss.backward(torch.tensor(1.0))
saved_grad=x.get_grad()

eps=gelib.SO3vec.randn_like(x)
z=gelib.CGproduct(x+eps,y)
lossd=z.odot(test_vec)
print(lossd-loss)
print(eps.odot(saved_grad))

print("\n\n")
