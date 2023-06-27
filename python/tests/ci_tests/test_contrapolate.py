import torch
import gelib as g
A=g.SO3partArr.randn(1,[5,5,5],2,2)
A.requires_grad_()
M=torch.randn(3,3,3,3)
B=A.conterpolate(M)
print(B.size())
print(B)

B.backward(B)
print(A.grad)


