import torch 
import cnine


print("\nDefining a 4x4 tensor of zeros:\n")

A=cnine.rtensor.zero([4,4])
print("A =")
print(A)

print("ndims =",A.ndims())
print("dim(0) =",A.dim(0))
dims=A.dims()
print("dims[0] =",dims[0])


print("\n\nManipulating tensor elements:\n")

A=cnine.rtensor.sequential(4,4)
print("A =")
print(A)

print("A(1,2) =",A([1,2]))
print("A(1,2) =",A[1,2])
print("A(1,2) =",A(1,2))
print("\n")

A[[1,2]]=99
print("A =")
print(A)


print("\nSimple tensor operations:\n")

B=cnine.rtensor.ones(4,4)

print("A+B =")
print(A+B)
print("A*5 =")
print(A*5)
print("A*A =")
print(A*A)

A+=B
print("A =")
print(A)
A-=B
print("A =")
print(A)


print("\nConverting from/to PyTorch tensors:\n")

A=torch.rand([3,3])
B=cnine.rtensor(A)
print("B =")
print(B)
B+=B
C=B.torch()
print("C =")
print(C)


print("\n\nFunctions of tensors:\n")

A=cnine.rtensor.randn(4,4)
B=cnine.rtensor.ones([4,4])
print("inp(A,B) =",cnine.inp(A,B),"\n")
print("norm2(A) =",cnine.norm2(A),"\n")
print("ReLU(A) =")
print(cnine.ReLU(A))


print("\nTransformations of tensors:\n")

A=cnine.rtensor.sequential(4,4)
print("A.transp() =")
print(A.transp())

print("A.slice(1,2) =")
print(A.slice(1,2))

print("A.reshape([2,8]) =")
print(A.reshape([2,8]))

print("A =")
print(A)


# --------------------------------------------------------

print("\nComplex tensors:\n")

A=cnine.ctensor.gaussian(4,4)
print("A =")
print(A)

print("A(1,2) =",A(1,2))
A[1,2]=3+4j
print("A =")
print(A)

print("A.conj() =")
print(A.conj())

B=A.torch()
print("B =")
print(B)

A=torch.rand([3,3,2])
print("A =")
print(A)

B=cnine.ctensor(A)
print("B =")
print(B)


# --------------------------------------------------------


print("\nTensor arrays:\n")

A=cnine.rtensorArr.gaussian([2,2],[4,4])
print("A =")
print(A)

adims=A.get_adims()
print("adims =",adims)
cdims=A.get_cdims()
print("cdims =",cdims)

B=A([0,1])
print("B =")
print(B)


A[[0,1]]=A([0,0])
print("A =")
print(A)


print("\nConversions to/from PyTorch:\n")

A=cnine.rtensorArr.sequential([2,2],[3,3])
print("A =")
print(A)

B=A.torch()
print("B =",B)
print(B)


A=torch.rand([2,3,3])
print("A =")
print(A)


B=cnine.rtensorArr(1,A)
print("B =",B)
print(B)


print("\nOperations:\n")

A=cnine.rtensorArr.zero([2,2],[3,3])
B=cnine.rtensorArr.ones([2,2],[3,3])
C=A+B
print("C([0,1] =")
print(C([0,1]))

A=cnine.rtensorArr.zero([2,2],[3,3])
B=cnine.rtensor.ones([3,3])
C=A+B
print("C =")
print(C)


A=cnine.rtensorArr.gaussian([2,2],[4,4])
B=A.reduce(1)
print("B =")
print(B)


C=B.widen(1,3)
print("C =")
print(C)




