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

    def __str__(self):
        return _SO3part.view(self).__str__()

    def __repr__(self):
        return _SO3part.view(self).__repr__()



class CGproduct(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y, l):
        #u=GElib.CGproduct(_SO3part.view(x),_SO3part.view(y),l)
        #return SO3part(u.torch())
        return SO3part(GElib.CGproduct(_SO3part.view(x),_SO3part.view(y)).torch())
