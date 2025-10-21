import torch
import gelib as g

tau1=g.SO3type({0:2,1:3})
tau2=g.SO3type({0:3,1:4})
print(tau1)
print(tau2)

v=g.SO3vec.randn(1,tau1)
print(v)

W=g.SO3weights.randn(tau1,tau2)
print(W)

u=v*W
print(u)


