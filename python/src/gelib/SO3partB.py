# This file is part of GElib, a C++/CUDA library for group
# equivariant tensor operations. 
# 
# Copyright (c) 2022, Imre Risi Kondor
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch
from cnine import ctensorb 
from gelib_base import SO3partB as _SO3partB


# ----------------------------------------------------------------------------------------------------------
# ---- SO3part ---------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3partB(torch.Tensor):
    """
    A collection of vectors that transform according to a specific irreducible representation of SO(3).
    The vectors are stacked into a third order tensor. The first index is the batch index, the second
    is m=-l,...,l, and the third index is the fragment index. 
    """

    def __init__(self):
        self.obj=_SO3partB.zero(1,1,1)

    def __init__(self, x):
        if(isinstance(x,SO3partB)):
            super().__init__(x)
            self.obj=x.obj
            return
        if(isinstance(x,_SO3partB)):
            self.obj=x
            return
        if(isinstance(x,torch.Tensor)):
           self=SO3partB_InitFromTorchTensorFn.apply(x)
       
    @classmethod
    def raw(self, b, l, n, _dev=0):
        r=SO3partB([1])
        r.obj=_SO3partB.raw(b,l,n,_dev)
        return r

    @classmethod
    def zeros(self, b, l, n, _dev=0):
        r=SO3partB([1])
        r.obj=_SO3partB.zero(b,l,n,_dev)
        return r

    @classmethod
    def randn(self, b, l, n, _dev=0):
        r=SO3partB([1])
        r.obj=_SO3partB.gaussian(b,l,n,_dev)
        return r

    @classmethod
    def Fraw(self, b, l, _dev=0):
        r=SO3partB([1])
        r.obj=_SO3partB.Fraw(b,l,_dev)
        return r

    @classmethod
    def Fzeros(self, b, l, n, _dev=0):
        r=SO3partB([1])
        r.obj=_SO3partB.Fzero(b,l,n,_dev)
        return r

    @classmethod
    def Frandn(self, b, l, n, _dev=0):
        r=SO3partB([1])
        r.obj=_SO3partB.Fgaussian(b,l,n,_dev)
        return r

    @classmethod
    def spharm(self, l, X):
        x=_SO3partB.zero(X.size(0),l,X.size(2))
        x.add_spharm(X)
        return SO3partB(x)

    @classmethod
    def zeros_like(self,x):
        r=SO3partB([1])
        r.obj=_SO3partB.zeros_like(x.obj)
        return r



    # -------------------------------------------------------------------------------------------------------


    def get_dev(self):
        return self.obj.get_dev()

    def getb(self):
        return self.obj.getb()

    def getl(self):
        return self.obj.getl()

    def getn(self):
        return self.obj.getn()

    def _get_grad(self):
        return self.obj.get_grad()
    
    def _view_of_grad(self):
        return self.obj.view_of_grad()
    
    def get_grad(self):
        R=SO3partB(1)
        R.obj=self.obj.get_grad()
        return R
    
    def view_of_grad(self):
        R=SO3partB(1)
        R.obj=self.obj.view_of_grad()
        return R
    
    def add_to_grad(self,x):
        R=SO3partB(1)
        R.obj=self.obj.add_to_grad(x.obj.get_grad())
        return R

    def torch(self):
        return SO3partB_ToTorchTensorFn.apply(self)


    # -------------------------------------------------------------------------------------------------------


    def __add__(self,y):
        print("add")
        r=SO3partB(1)
        r.obj=self.obj.plus(y)
        return r
        
    def __radd__(self,y):
        print("radd")
        self.obj.add(y)
        

    def __mult__(self,y):
        return SO3partB_timesCtensorFn.apply(self,y)
    
    def inp(self,y):
        return SO3partB_inpFn.apply(self,y)

    
    # ---- CG-products ---------------------------------------------------------------------------------------


    def CGproduct(self, y, l):
        """
        Compute the l component of the Clesbsch--Gordan product of this SO3part with another SO3part y.
        """
        return SO3partB_CGproductFn.apply(self,y,l)

    def DiagCGproduct(self, y, l):
        """
        Compute the l component of the diagonal Clesbsch--Gordan product of this SO3part with another SO3part y.
        """
        return SO3partB_DiagCGproductFn.apply(self,y,l)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ----------------------------------------------------------------------------------------------------------
# ---- Autograd functions -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3partB_CGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l):
        r=SO3partB.zeros(x.getb(),l,x.getn()*y.getn(),x.get_dev())
        r.obj.addCGproduct(x.obj,y.obj)
        ctx.x=x
        ctx.y=y
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx, g):
        xg=SO3partB(1)
        yg=SO3partB(1)
        ctx.x._view_of_grad().addCGproduct_back0(ctx.r._get_grad(),ctx.y.obj)
        ctx.y._view_of_grad().addCGproduct_back1(ctx.r._get_grad(),ctx.x.obj)
        return xg,yg,None


class SO3partB_DiagCGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l):
        assert x.getn()==y.getn()
        r=SO3partB.zeros(x.getb(),l,x.getn(),x.get_dev())
        r.obj.addDiagCGproduct(x.obj,y.obj)
        ctx.x=x
        ctx.y=y
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx, g):
        xg=SO3partB(1)
        yg=SO3partB(1)
        ctx.x._view_of_grad().addDiagCGproduct_back0(ctx.r._get_grad(),ctx.y.obj)
        ctx.y._view_of_grad().addDiagCGproduct_back1(ctx.r._get_grad(),ctx.x.obj)
        return xg,yg,None


class SO3partB_InitFromTorchTensorFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        assert(x.dim()==4)
        assert(x.size(3)==2)
        assert(x.size(1)%2==1)
        r=SO3partB(1)
        r.obj=_SO3partB(x.obj)
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx,g):
        assert ctx.r != None
        return ctx.r.view_of_grad().torch()


class SO3partB_ToTorchTensorFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        return x.obj.torch()

    @staticmethod
    def backward(ctx,g):
        assert(g.dim()==4)
        assert(g.size(3)==2)
        assert(g.size(1)%2==1)

        assert ctx.x != None
        return ctx.x.add_to_grad(_SO3partB(g))


class SO3partB_timesCtensorFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y):
        r=SO3partB(1)
        r.obj=x.mprod(y)
        ctx.x=x
        ctx.y=y
        ctx.r=r
        return r

    def backward(ctx,g):
        xg=SO3partB(1)
        yg=SO3partB(1)
        ctx.x._view_of_grad().add_mprod_back0(ctx.r._view_of_grad(),ctx.y.obj)
        ctx.x._view_of_grad().add_mprod_back1(ctx.r._view_of_grad(),ctx.x.obj)
        return xg,yg

    
class SO3partB_inpFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y):
        r=ctensorb(1)
        r.obj=x.mprod(y)
        ctx.x=x
        ctx.y=y
        ctx.r=r
        return r

    def backward(ctx,g):
        xg=SO3partB(1)
        yg=SO3partB(1)
        ctx.x._view_of_grad().add_mprod_back0(ctx.r._view_of_grad(),ctx.y.obj)
        ctx.x._view_of_grad().add_mprod_back1(ctx.r._view_of_grad(),ctx.x.obj)
        return xg,yg

    
# ----------------------------------------------------------------------------------------------------------
# ---- Other functions --------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


#def CGproduct(x, y, maxl=-1):
#    return x.CGproduct(y, maxl)


#def DiagCGproduct(x, y, maxl=-1):
#    return x.DiagCGproduct(y, maxl)


