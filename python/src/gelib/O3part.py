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
# ---- O3part ---------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class O3part(torch.Tensor):
    """
    A collection of vectors that transform according to a specific irreducible representation of O(3).
    The vectors are stacked into a third order tensor. The first index is the batch index, the second
    is m=-l,...,l, and the third index is the fragment index. 
    """

    def __new__(cls,T,mu):
        assert isinstance(mu,tuple)
        assert len(mu)==2
        assert isinstance(mu[0],int)
        assert isinstance(mu[1],int)
        R=super().__new__(O3part,T)
        R.mu=mu
        return R

    def __init__(self, _T, p):
        self=_T
        #super().__init__()


    # ---- Static constructors -----------------------------------------------------------------------------


    @classmethod
    def zeros(self,b,mu,n,device='cpu'):
        """
        Create an O(3)-part consisting of b lots of n vectors transforming according to the l'th irrep of O(3).
        The vectors are initialized to zero, resulting in an b*(2+l+1)*n dimensional complex tensor of zeros.
        """
        return O3part(torch.zeros([b,2*mu[0]+1,n],dtype=torch.complex64,device=device),mu)

    @classmethod
    def randn(self,b,mu,n,device='cpu'):
        """
        Create an O(3)-part consisting of b lots of n vectors transforming according to the l'th irrep of O(3).
        The vectors are initialized as random gaussian vectors, resulting in an b*(2+l+1)*n dimensional random
        complex tensor.
        """
        return O3part(torch.randn([b,2*mu[0]+1,n],dtype=torch.complex64,device=device),mu)

    @classmethod
    def spharm(self,mu,X,device='cpu'):
        """
        Return the spherical harmonics of the vectors in the tensor (x,y,z)
        """
        assert(X.dim()==3)
        R = O3part.zeros(X.size(0),mu,X.size(2),device=device)
        R.backend().add_spharm(X)
        return R

    @classmethod
    def Fzeros(self,b,mu,device='cpu'):
        """
        Create an O(3)-part corresponding to the l'th matrix in the Fourier transform of a function on O(3).
        This gives a b*(2+l+1)*(2l+1) dimensional complex tensor. 
        """
        return O3part(torch.zeros([b,2*mu[0]+1,2*l+1],dtype=torch.complex64,device=device),mu)

    @classmethod
    def Frandn(self,b,mu,device='cpu'):
        """
        Create an O(3)-part corresponding to the l'th matrix in the Fourier transform of a function on O(3).
        This gives a b*(2+l+1)*(2l+1) dimensional complex random tensor. 
        """
        return O3part(torch.randn([b,2*mu[0]+1,2*l+1],dtype=torch.complex64,device=device),mu)

    def zeros_like(self,*args):
        if not args:
            return O3part(torch.zeros_like(self),self.mu)
        if len(args)==2:
            assert isinstance(args[0],int)
            assert isinstance(args[1],int)
            dims=list(self.size())
            dims[-2]=2*args[0]+1
            dims[-1]=args[1]
            return O3part(torch.zeros(dims,dtype=torch.complex64,device=self.device),self.mu)
        if len(args)==3:
            assert isinstance(args[0],int)
            assert isinstance(args[1],int)
            dims=list(self.size())
            dims[0]=args[0]
            dims[-2]=2*args[1]+1
            dims[-1]=args[2]
            return O3part(torch.zeros(dims,dtype=torch.complex64,device=self.device),self.mu)
    
#     @classmethod
#     def randn_like(self):
#         """
#         Create an SO(3)-part consisting of b lots of n vectors transforming according to the l'th irrep of SO(3).
#         The vectors are initialized as random gaussian vectors, resulting in an b*(2+l+1)*n dimensional random
#         complex tensor.
#         """
#         return O3part(torch.randn(self.size(),dtype=torch.complex64,device=self.device))

    @classmethod
    def randn_like(self,x):
        return O3part(torch.randn_like(x),self.mu)

    def backend(self):
        return gb.O3part.view(self,self.mu)

                   
    # ---- Access ------------------------------------------------------------------------------------------


    def getb(self):
        return self.size(0)

    def getp(self):
        return self.mu[1]

    def getl(self):
        return self.mu[0]
        #return int((self.size(1)-1)/2)

    def getn(self):
        return self.size(2)


    # ---- Operations --------------------------------------------------------------------------------------


    def apply(self, R):
        assert(isinstance(R,O3element))
        rho=O3irrep(self.getl(),self.getp())
        return O3part(torch.matmul(rho.matrix(R),self),self.mu)


    # ---- Products -----------------------------------------------------------------------------------------


    def odot(self,y):
            return torch.Tensor(torch.sum(torch.mul(torch.view_as_real(self),torch.view_as_real(y))))

    def CGproduct(self,y,mu):
        """
        Compute the l component of the Clesbsch--Gordan product of this O3part with another O3part y.
        """
        assert isinstance(y,O3part)
        return O3part_CGproductFn.apply(self,y,mu)

    def DiagCGproduct(self,y,mu):
        """
        Compute the l component of the diagonal Clesbsch--Gordan product of this O3part with another O3part y.
        """
        assert isinstance(y,O3part)
        return O3part_DiagCGproductFn.apply(self,y,mu)


    # ---- I/O ----------------------------------------------------------------------------------------------

    def __repr__(self):
        return self.backend().__repr__()

    def __str__(self):
        return self.backend().__str__()


# ----------------------------------------------------------------------------------------------------------
# ---- Autograd functions -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class O3part_CGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,mu):
        assert isinstance(mu,tuple)
        assert isinstance(mu[0],int)
        assert isinstance(mu[1],int)
        assert x.getp()*y.getp()==mu[1]
        ctx.mu=mu
        ctx.save_for_backward(x,y)
        b=common_batch(x,y)
        r=O3part.zeros(b,mu,x.size(-1)*y.size(-1),device=x.device)
        r.backend().add_CGproduct(x.backend(),y.backend())
        return r

    @staticmethod
    def backward(ctx,g):
        x,y = ctx.saved_tensors
        xg=x.zeros_like()
        yg=y.zeros_like()
        xg.backend().add_CGproduct_back0(gb.O3part.view(g,ctx.mu[1]),y.backend())
        yg.backend().add_CGproduct_back1(gb.O3part.view(g,ctx.mu[1]),x.backend())
        return xg,yg,None


class O3part_DiagCGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,mu):
        assert isinstance(mu,tuple)
        assert isinstance(mu[0],int)
        assert isinstance(mu[1],int)
        assert x.getp()*y.getp()==mu[1]
        ctx.save_for_backward(x,y)
        b=common_batch(x,y)
        assert x.size(-1)==y.size(-1)
        r=O3part.zeros(b,mu,x.size(-1),device=x.device)
        r.backend().add_DiagCGproduct(x.backend(),y.backend())
        return r

    @staticmethod
    def backward(ctx,g):
        x,y = ctx.saved_tensors
        xg=x.zeros_like()
        yg=y.zeros_like()
        xg.backend().add_DiagCGproduct_back0(gb.O3part.view(g,ctx.mu[1]),y.backend())
        yg.backend().add_DiagCGproduct_back1(gb.O3part.view(g,ctx.mu[1]),x.backend())
        return xg,yg,None


