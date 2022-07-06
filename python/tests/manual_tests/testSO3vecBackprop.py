import torch
import gelib 


b=1
tau=[1,1]

# Define two random SO3vec objects  
x=gelib.SO3vec.randn(b,tau)
y=gelib.SO3vec.randn(b,tau)
x.parts[1].requires_grad_()

# Compute the CG-product
z=gelib.CGproduct(x,y)

print("CG-product:")
print(z)

#loss=torch.norm(z.parts[2])
loss=torch.view_as_real(z.parts[1])[0,1,1,0]
print(loss)
loss.backward(torch.tensor(1.0))
saved_grad=x.parts[1].grad

eps=torch.randn([b,3,1],dtype=torch.cfloat)
x.parts[1]=x.parts[1]+eps

z=gelib.CGproduct(x,y)
print(z)
lossd=torch.view_as_real(z.parts[1])[0,1,1,0]
#lossd=torch.norm(z.parts[2])
print(lossd-loss)

print(torch.sum(torch.mul(torch.view_as_real(eps),torch.view_as_real(saved_grad))))

print("\n\n")
