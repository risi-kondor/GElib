from importlib.machinery import SourceFileLoader

import torch
import GElib
from GElib import SO3part as _SO3part
from GElib import SO3vec as _SO3vec

# TODO: THIS LINE SHOULD BE CHANGED! WE SHOULDN'T BE IMPORTING A PACKAGE USING ITS FILE PATH!
# I'VE FIXED IT SO IT RUNS (BEFORE IT WOULD ONLY RUN FROM ONE DIRECTORY) BUT IT SHOULD STILL CHANGE.
import os
real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)
print(__file__)
print(real_path)
print(dir_path)

# tcn= SourceFileLoader("torch_cnine", "../../../../cnine/python/torch_cnine.py").load_module()
tcn = SourceFileLoader("torch_cnine", dir_path + "/../../../../cnine/python/torch_cnine.py").load_module()




class SO3part(tcn.ctensor):

    @staticmethod
    def zeros(l,n):
        r=SO3part(torch.zeros(2,2*l+1,n))
        return r
    
    @staticmethod
    def ones(l,n):
        r=SO3part(torch.zeros(2,2*l+1,n))
        return r
    
    @staticmethod
    def randn(l,n):
        r=SO3part(torch.randn(2,2*l+1,n))
        return r


    def cview(self):
        return _SO3part.view(self)
        
    def CGproduct(self,y):
        if isinstance(y,SO3part):
            return SO3partCGproductFunction.apply(self,y)
        else:
            raise TypeError('Type of second argument in CGproduct(SO3part,SO3part) is not SO3part.')

        
    def __str__(self):
        return _SO3part.view(self).__str__()

    def __repr__(self):
        return _SO3part.view(self).__repr__()



class SO3partCGproductFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y, l):
        ctx.save_for_backward(x,y);
        return SO3part(GElib.SO3partCGproduct(_SO3part.view(x),_SO3part.view(y)).torch())

    @staticmethod
    def backward(ctx, grad):
        x,y=ctx.saved_tensors
        grad_x=grad_y=None
        if ctx.needs_input_grad[0]:
            grad_x=torch.zeros_like(x)
            grad_x.cview().SO3partCGproduct0(grad.cview(),y.cview())
        if ctx.needs_input_grad[1]:
            grad_y=torch.zeros_like(y)
            grad_y.cview().SO3partCGproduct1(grad.cview(),x.cview())



class tensorlike_plus_fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x,y)
        return x.constr(x.obj+y.obj)

    @staticmethod
    def backward(ctx, grad):
        x,y=ctx.saved_tensors
        grad_x=grad_y=None
        if ctx.needs_input_grad[0]:
            grad_x=grad
        if ctx.needs_input_grad[1]:
            grad_y=grad



class tensorlike(torch.Tensor):

    @staticmethod
    def __new__(cls,_obj):
        return super().__new__(cls)

    def __add__(self,y):
        return tensorlike_plus_fn.apply(self,y)



class SO3vec(tensorlike):

    def __init__(self,_obj):
        self.obj=_obj

    @staticmethod
    def constr(x):
        return SO3vec(x)

    @staticmethod
    def zeros(_tau):
        return SO3vec(_SO3vec.zero(_tau))

    @staticmethod
    def ones(_tau):
        return SO3vec(_SO3vec.ones(_tau))

    @staticmethod
    def gaussian(_tau):
        return SO3vec(_SO3vec.gaussian(_tau))


    def __str__(self):
        return self.obj.__str__()

    def __repr__(self):
        return self.obj.__repr__()

    

def CGproduct(x,y):
    return x.CGproduct(y)
