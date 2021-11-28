from importlib.machinery import SourceFileLoader

import torch

tcn= SourceFileLoader("torch_cnine", "../../cnine/python/torch_cnine.py").load_module()

import GElib
from GElib import SO3part as _SO3part


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
    

def CGproduct(x,y):
    return x.CGproduct(y)
