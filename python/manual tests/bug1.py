import torch
import gelib as g


# Create new tensors
zeros_a = torch.zeros((1,2,3,1,2))
zeros_b = torch.zeros((1,2,3,3,4))
zeros_1 = torch.complex(torch.randn_like(zeros_a), torch.zeros_like(zeros_a))
zeros_2 = torch.complex(torch.randn_like(zeros_b), torch.zeros_like(zeros_b))

# Perform a permutation
zeros_1 = zeros_1.permute((-3, -2, -1, 0, 1))
zeros_2 = zeros_2.permute((-3, -2, -1, 0, 1))

# Force the Tensors to create a new instance instead of just a view.
zeros_1 = torch.stack([zeros_1[0,...],zeros_1[1,...],zeros_1[1,...],zeros_1[2,...],zeros_1[2,...]], dim = 0)
zeros_2 = torch.stack([zeros_2[0,...],zeros_2[1,...],zeros_2[1,...],zeros_2[2,...],zeros_2[2,...]], dim = 0)

# Modify the order of the new instance as a view.
zeros_1 = zeros_1.permute((3, 4, 0, 1, 2))
zeros_2 = zeros_2.permute((3, 4, 0, 1, 2))

# Make new tensors in shape matching the above view.
zeros_3 = torch.complex(torch.zeros_like(zeros_1.real), torch.zeros_like(zeros_1.real))
zeros_4 = torch.complex(torch.zeros_like(zeros_2.real), torch.zeros_like(zeros_2.real))

p1=g.SO3partArr(zeros_1)
p2=g.SO3partArr(zeros_2)
p3=g.SO3partArr(zeros_3)
p4=g.SO3partArr(zeros_4)

print(p1.size())
print(p3.size())
print(p1.getl())
print(p3.getl())

p1.contiguous()
p3.contiguous()

p1.CGproduct(p3,0)
print("aaa")

# Put it into SO3vecArr instances and take CG product.
arr1 = g.SO3vecArr(g.SO3partArr(zeros_1), g.SO3partArr(zeros_2))
arr2 = g.SO3vecArr(g.SO3partArr(zeros_3), g.SO3partArr(zeros_4))

arr1.CGproduct(arr2)
