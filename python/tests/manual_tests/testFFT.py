import gelib

dev=0

v=gelib.SO3vec.Frandn(1,6,dev)        # define a random SO3-vector
v.parts[2].requires_grad_()           # make sure that we can backprop

r=v.iFFT(20)                          # compute an inverse Fourier transform with resolution 20
print(r.size())                       # see how big it is

r.backward(r)                         # backprop something
print(v.parts[2].grad/v.parts[2])    # should be more or less constant
