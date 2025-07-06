# This file is part of GElib, a C++/CUDA library for group equivariant 
# tensor operations. 
#  
# Copyright (c) 2025, Imre Risi Kondor
#
# This source code file is subject to the terms of the noncommercial 
# license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
# use is prohibited. All redistributed versions of this file (in orginal
# or modified form) must retain this copyright notice and must be 
# accompanied by a verbatim copy of the license. 

import torch
import gelib_base as gb
from gelib import *


# ----------------------------------------------------------------------------------------------------------
# ---- O3vec ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class O3vec:
    """
    An O(3)-covariant vector consisting of a sequence of O3part objects, each transforming according
    to a specific irrep of O(3).
    """

    def __init__(self,*args):
        self.parts={}
        if not args:
            return
        for x in args:
            assert isinstance(x,torch.Tensor)
            p=O3part(x)
            self.parts[p.getl()]=x
            

    # ---- Static constructors ------------------------------------------------------------------------------


    @classmethod
    def zeros(self,b,tau,device='cpu'):
        "Construct a zero O3vec object of given type _tau."
        R=O3vec()
        if isinstance(tau,dict):
            for mu,n in tau.items():
                R.parts[mu]=O3part.zeros(b,mu,n,device=device)
        return R

    @classmethod
    def randn(self,b,tau,device='cpu'):
        "Construct a random O3vec object of given type _tau."
        R=O3vec()
        if isinstance(tau,dict):
            for mu,n in tau.items():
                R.parts[mu]=O3part.randn(b,mu,n,device=device)
        return R

    @classmethod
    def Fzeros(self,b,tau,device='cpu'):
        "Construct a zero O3vec object of given type _tau."
        R=O3vec()
        if isinstance(tau,dict):
            for mu,n in tau.items():
                R.parts[mu]=O3part.Fzeros(b,mu,n,device=device)
        return R

    @classmethod
    def Frandn(self,b,tau,device='cpu'):
        "Construct a zero O3vec object of given type _tau."
        R=O3vec()
        if isinstance(tau,dict):
            for mu,n in tau.items():
                R.parts[mu]=O3part.Frandn(b,mu,n,device=device)
        return R

    @classmethod
    def spharm(self,mu,X,device='cpu'):
        """
        Return the spherical harmonics of the vectors in the tensor X
        """
        assert(X.dim()==3)
        R=O3vec()
        R.parts[mu]=O3part.spharm(b,mu,X,device=device)
        return R

    @classmethod
    def zeros_like(self, x):
        R=O3vec()
        for mu,p in x.parts.items():
            R.parts[mu]=O3part.zeros_like(p)
        return R

    @classmethod
    def randn_like(self, x):
        R=O3vec()
        for l,p in x.parts.items():
            R.parts[mu]=O3part.randn_like(p)
        return R

    def backend(self):
        return gb.O3vec.view(self.parts)


    # ---- Access -------------------------------------------------------------------------------------------


    def getb(self):
        return self.parts[min(self.parts)].getb()

    def tau(self):
        "Return the 'type' of the O3vec."
        r={}
        for l,p in self.parts.items():
            r[l]=p.get_mu()
        return r

    def get_type(self):
        "Return the 'type' of the O3vec, i.e.."
        r={}
        for l,p in self.parts.items():
            r[l]=p.get_mu()
        return r

    def requires_grad_(self):
        for l,p in self.parts.items():
            p.requires_grad_()

    def get_grad(self):
        return O3vec(*[p.grad for l,p in self.parts.items()])


    # ---- Operations ---------------------------------------------------------------------------------------


    def apply(self, R):
        "Apply the group element to this vector"
        r = O3vec()
        for l,p in self.parts.items():
            r.parts[l]=p.apply(R)
        return r

    def odot(self,y):
        assert(list(self.parts.keys())==list(y.parts.keys()))
        return sum([self.parts[l].odot(y.parts[l]) for l in self.parts.keys()])

    def __add__(self,y):
        assert(list(self.parts.keys())==list(y.parts.keys()))
        return O3vec(*[self.parts[l]+y.parts[l] for l in self.parts.keys()])

        
    # ---- Products -----------------------------------------------------------------------------------------


    def CGproduct(self, y, maxl=-1):
        """
        Compute the full Clesbsch--Gordan product of this O3vec with another O3vec y.
        """
        xparts=list(self.parts.values())
        yparts=list(y.parts.values())
        rparts =list(O3vec_CGproductFn.apply(len(xparts), len(yparts), maxl,*(xparts+yparts)))
        return O3vec(*rparts)

    def DiagCGproduct(self, y, maxl=-1):
        """
        Compute the diagonal Clesbsch--Gordan product of this O3vec with another O3vec y.
        """
        xparts=list(self.parts.values())
        yparts=list(y.parts.values())
        rparts =list(O3vec_DiagCGproductFn.apply(len(xparts), len(yparts), maxl,*(xparts+yparts)))
        return O3vec(*rparts)


    # ---- I/O ----------------------------------------------------------------------------------------------

    def __repr__(self):
        return self.backend().__repr__()

    def __str__(self):
        return self.backend().__str__()


# ----------------------------------------------------------------------------------------------------------
# ---- Autograd functions -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class O3vec_CGproductFn(torch.autograd.Function):

    def __init__(self):
        self.is_sparse=False

    @staticmethod
    def forward(ctx, k1, k2, maxl, *args):
        ctx.k1 = k1
        ctx.k2 = k2
        ctx.save_for_backward(*args)

        x=gb.O3vec.view(args[0:k1])
        y=gb.O3vec.view(args[k1:k1+k2])
        b=common_batch(args[0],args[k1])
        tau=x.get_tau().CGproduct(y.get_tau(),maxl)
        rparts=MakeZeroO3parts(b,tau.get_parts(),args[0].device) # just use regular constructor?
        r=gb.O3vec.view(rparts)
        r.addCGproduct(x,y)

        return tuple(rparts)


    @staticmethod
    def backward(ctx, *args):

        k1 = ctx.k1
        k2 = ctx.k2
        grads=[torch.zeros_like(x) for x in ctx.saved_tensors]

        x=gb.O3vec.view(ctx.saved_tensors[0:k1])
        y=gb.O3vec.view(ctx.saved_tensors[k1:k1+k2])
        g=gb.O3vec.view(args)
        xg=gb.O3vec.view(grads[0:k1])
        yg=gb.O3vec.view(grads[k1:k1+k2])
        xg.addCGproduct_back0(g,y)
        yg.addCGproduct_back1(g,x)

        return tuple([None,None,None]+grads)


class O3vec_DiagCGproductFn(torch.autograd.Function):

    def __init__(self):
        self.is_sparse=False

    @staticmethod
    def forward(ctx, k1, k2, maxl, *args):
        ctx.k1 = k1
        ctx.k2 = k2
        ctx.save_for_backward(*args)

        x=gb.O3vec.view(args[0:k1])
        y=gb.O3vec.view(args[k1:k1+k2])
        b=common_batch(args[0],args[k1])
        tau=x.get_tau().DiagCGproduct(y.get_tau(),maxl)
        rparts=MakeZeroO3parts(b,tau.get_parts(),args[0].device)
        r=gb.O3vec.view(rparts)
        r.addDiagCGproduct(x,y)

        return tuple(rparts)

    @staticmethod
    def backward(ctx, *args):

        k1 = ctx.k1
        k2 = ctx.k2
        grads=[torch.zeros_like(x) for x in ctx.saved_tensors]

        x=gb.O3vec.view(ctx.saved_tensors[0:k1])
        y=gb.O3vec.view(ctx.saved_tensors[k1:k1+k2])
        g=gb.O3vec.view(args)
        xg=gb.O3vec.view(grads[0:k1])
        yg=gb.O3vec.view(grads[k1:k1+k2])
        xg.addDiagCGproduct_back0(g,y)
        yg.addDiagCGproduct_back1(g,x)

        return tuple([None,None,None]+grads)









# ----------------------------------------------------------------------------------------------------------
# ---- Other functions --------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


#def CGproduct(x, y, maxl=-1):
#    return x.CGproduct(y, maxl)


#def DiagCGproduct(x, y, maxl=-1):
#    return x.DiagCGproduct(y, maxl)


#def Fproduct(x, y, maxl=-1):
#    return x.Fproduct(y, maxl)


#def Fmodsq(x, a=-1):
#    return x.Fmodsq(a)


# ---- Helpers -----------------------------------------------------------------------------------------------


# def tau_type(x):
#     r = []
#     for t in x:
#         r.append(t.size(2))
#     return r


# def CGproductType(x, y, maxl=-1):
#     if maxl == -1:
#         maxl = len(x)+len(y)-2
#     maxl = min(maxl, len(x)+len(y)-2)
#     r = [0]*(maxl+1)
#     for l1 in range(0, len(x)):
#         for l2 in range(0, len(y)):
#             for l in range(abs(l1-l2), min(l1+l2, maxl)+1):
#                 r[l] += x[l1]*y[l2]
#     return r


# def DiagCGproductType(x, y, maxl=-1):
#     if maxl == -1:
#         maxl = len(x)+len(y)-2
#     maxl = min(maxl, len(x)+len(y)-2)
#     r = [0]*(maxl+1)
#     for l1 in range(0, len(x)):
#         for l2 in range(0, len(y)):
#             for l in range(abs(l1-l2), min(l1+l2, maxl)+1):
#                 r[l] += x[l1]
#     return r


# def DDiagCGproductType(x, maxl=-1):
#     if maxl == -1:
#         maxl = len(x)+(1-len(x)%2)-1
#     maxl = min(maxl, len(x)+(1-len(x)%2)-1)
#     r = [0]*(maxl+1)
#     for l in range(0, len(x)):
#         r[l+l%2] += x[l]
#     return r


def MakeZeroO3parts(b, tau, device):
    R=[]
    for l,n in tau.items():
        R.append(O3part.zeros(b,l,n,device))
        #R.append(torch.zeros([b,2*l+1,tau[l]],dtype=torch.cfloat,device=device))
    return R


def makeZeroO3Fparts(b, maxl, device):
    R = []
    for l,n in tau.items():
        R.append(O3part.Fzeros(b,l,device))
    return R


