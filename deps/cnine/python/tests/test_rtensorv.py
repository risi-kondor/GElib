import torch
import cnine

A=cnine.rtensorv.randn([4,4])
print(A)

u=torch.randn([3,3])
u.requires_grad_()
print(u)

A=cnine.rtensorv.init(u)
print(A)

#A.add_to_grad(A)
#A.backward(A)

B=A.torch()
print(B)

B.backward(B)
print(u.grad)


# -------------------------------------------------------------


A=cnine.rtensorv.randn([4,4])
A.requires_grad_()
print(A)

c=cnine.norm2(A)

print(c)

c.backward(torch.tensor([1]))
print(A.get_grad())


# -------------------------------------------------------------

print("\nInp:\n")

A=cnine.rtensorv.randn([4,4])
print(A)
B=cnine.rtensorv.randn([4,4])
print(B)
A.requires_grad_()

c=cnine.inp(A,B)

print(c)

c.backward(torch.tensor([1]))
print(A.get_grad())


# -------------------------------------------------------------

print("\nDiff2:\n")

A=cnine.rtensorv.randn([4,4])
print(A)
B=cnine.rtensorv.randn([4,4])
print(B)
A.requires_grad_()

c=cnine.diff2(A,B)

print(c)

c.backward(torch.tensor([1]))
print(A.get_grad())


