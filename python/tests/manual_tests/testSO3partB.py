import torch
import gelib
import torch.autograd

def hookfn(g):
    print("123")
    print(g.__repr__())
    return gelib.SO3partB.zeros(1,2,2)

u=gelib.SO3partB.randn(1,2,2)
u.requires_grad_()
u.register_hook(hookfn)

v=gelib.SO3partB.randn(1,2,2)
w=gelib.CGproduct(u,v,2)
#w.register_hook(hookfn)

#u.grad=v %u.detach()
#print(u.grad)

w.add_to_grad(w.obj)

w.backward(w)
print(1)
print(u.get_grad())
print(2)

#torch.autograd.grad(w,u,create_graph=True)

#print(w)

