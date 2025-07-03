# This file is part of GElib, a C++/CUDA library for group equivariant 
# tensor operations. 
#  
# Copyright (c) 2023, Imre Risi Kondor
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
# ---- SO3part ---------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3part(torch.Tensor):
    """
    A collection of vectors that transform according to a specific irreducible representation of SO(3).
    The vectors are stacked into a third order tensor. The first index is the batch index, the second
    is m=-l,...,l, and the third index is the fragment index. 
    """

    def __init__(self, _T):
        self=_T
        #super().__init__()


    # ---- Static constructors -----------------------------------------------------------------------------


    @classmethod
    def zeros(self,b,l,n,device='cpu'):
        """
        Create an SO(3)-part consisting of b lots of n vectors transforming according to the l'th irrep of SO(3).
        The vectors are initialized to zero, resulting in an b*(2+l+1)*n dimensional complex tensor of zeros.
        """
        return SO3part(torch.zeros([b,2*l+1,n],dtype=torch.complex64,device=device))

    @classmethod
    def randn(self,b,l,n,device='cpu'):
        """
        Create an SO(3)-part consisting of b lots of n vectors transforming according to the l'th irrep of SO(3).
        The vectors are initialized as random gaussian vectors, resulting in an b*(2+l+1)*n dimensional random
        complex tensor.
        """
        return SO3part(torch.randn([b,2*l+1,n],dtype=torch.complex64,device=device))

    @classmethod
    def spharm(self,l,X,device='cpu'):
        """
        Return the spherical harmonics of the vectors in the tensor (x,y,z)
        """
        assert(X.dim()==3)
        R = SO3part.zeros(X.size(0),l,X.size(2),device=device)
        R.backend().add_spharm(X)
        return R

    @classmethod
    def Fzeros(self,b,l,device='cpu'):
        """
        Create an SO(3)-part corresponding to the l'th matrix in the Fourier transform of a function on SO(3).
        This gives a b*(2+l+1)*(2l+1) dimensional complex tensor. 
        """
        return SO3part(torch.zeros([b,2*l+1,2*l+1],dtype=torch.complex64,device=device))

    @classmethod
    def Frandn(self,b,l,device='cpu'):
        """
        Create an SO(3)-part corresponding to the l'th matrix in the Fourier transform of a function on SO(3).
        This gives a b*(2+l+1)*(2l+1) dimensional complex random tensor. 
        """
        return SO3part(torch.randn([b,2*l+1,2*l+1],dtype=torch.complex64,device=device))

    def zeros_like(self,*args):
        if not args:
            return SO3part(torch.zeros_like(self))
        if len(args)==2:
            assert isinstance(args[0],int)
            assert isinstance(args[1],int)
            dims=list(self.size())
            dims[-2]=2*args[0]+1
            dims[-1]=args[1]
            return SO3part(torch.zeros(dims,dtype=torch.complex64,device=self.device))
        if len(args)==3:
            assert isinstance(args[0],int)
            assert isinstance(args[1],int)
            dims=list(self.size())
            dims[0]=args[0]
            dims[-2]=2*args[1]+1
            dims[-1]=args[2]
            return SO3part(torch.zeros(dims,dtype=torch.complex64,device=self.device))
    
#     @classmethod
#     def randn_like(self):
#         """
#         Create an SO(3)-part consisting of b lots of n vectors transforming according to the l'th irrep of SO(3).
#         The vectors are initialized as random gaussian vectors, resulting in an b*(2+l+1)*n dimensional random
#         complex tensor.
#         """
#         return SO3part(torch.randn(self.size(),dtype=torch.complex64,device=self.device))

    @classmethod
    def randn_like(self,x):
        return SO3part(torch.randn_like(x))

    def backend(self):
        return gb.SO3part.view(self)

                   
    # ---- Access ------------------------------------------------------------------------------------------


    def getb(self):
        return self.size(0)

    def getl(self):
        return int((self.size(1)-1)/2)

    def getn(self):
        return self.size(2)


    # ---- Operations --------------------------------------------------------------------------------------


    def apply(self, R):
        assert(isinstance(R,SO3element))
        rho=SO3irrep(self.getl())
        return SO3part(torch.matmul(rho.matrix(R),self))


    # ---- Products -----------------------------------------------------------------------------------------


    def odot(self,y):
            return torch.Tensor(torch.sum(torch.mul(torch.view_as_real(self),torch.view_as_real(y))))

    def CGproduct(self, y, l):
        """
        Compute the l component of the Clesbsch--Gordan product of this SO3part with another SO3part y.
        """
        assert isinstance(y,SO3part)
        assert isinstance(l,int)
        return SO3part_CGproductFn.apply(self,y,l)

    def DiagCGproduct(self, y, l):
        """
        Compute the l component of the diagonal Clesbsch--Gordan product of this SO3part with another SO3part y.
        """
        assert isinstance(y,SO3part)
        assert isinstance(l,int)
        return SO3part_DiagCGproductFn.apply(self,y,l)


    # ---- I/O ----------------------------------------------------------------------------------------------

    def __repr__(self):
        return self.backend().__repr__()

    def __str__(self):
        return self.backend().__str__()


# ----------------------------------------------------------------------------------------------------------
# ---- Autograd functions -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3part_CGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l):
        ctx.l=l
        ctx.save_for_backward(x,y)
        #b=max(x.size(0),y.size(0))
        #r=x.zeros_like(l,x.size(-1)*y.size(-1))
        b=common_batch(x,y)
        r=SO3part.zeros(b,l,x.size(-1)*y.size(-1),device=x.device)
        r.backend().add_CGproduct(x.backend(),y.backend())
        return r

    @staticmethod
    def backward(ctx,g):
        x,y = ctx.saved_tensors
        #g=SO3part(_g)
        xg=x.zeros_like()
        yg=y.zeros_like()
        gb.SO3part.view(xg).add_CGproduct_back0(gb.SO3part.view(g),gb.SO3part.view(y))
        gb.SO3part.view(yg).add_CGproduct_back1(gb.SO3part.view(g),gb.SO3part.view(x))
        return xg,yg,None


class SO3part_DiagCGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l):
        ctx.l=l
        ctx.save_for_backward(x,y)
        #b=max(x.size(0),y.size(0))
        #r=x.zeros_like(l,x.size(-1))
        b=common_batch(x,y)
        assert x.size(-1)==y.size(-1)
        r=SO3part.zeros(b,l,x.size(-1),device=x.device)
        r.backend().add_DiagCGproduct(x.backend(),y.backend())
        return r

    @staticmethod
    def backward(ctx,g):
        x,y = ctx.saved_tensors
        xg=x.zeros_like()
        yg=y.zeros_like()
        gb.SO3part.view(xg).add_DiagCGproduct_back0(gb.SO3part.view(g),gb.SO3part.view(y))
        gb.SO3part.view(yg).add_DiagCGproduct_back1(gb.SO3part.view(g),gb.SO3part.view(x))
        return xg,yg,None


# ----------------------------------------------------------------------------------------------------------
# ---- Other functions --------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


#def CGproduct(x, y, maxl=-1):
#    return x.CGproduct(y, maxl)


#def DiagCGproduct(x, y, maxl=-1):
#    return x.DiagCGproduct(y, maxl)


