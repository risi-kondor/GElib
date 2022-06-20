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
from gelib_base import SO3vecB as _SO3vecB


# ----------------------------------------------------------------------------------------------------------
# ---- SO3part ---------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3vecB(torch.Tensor):
    """
    An SO(3)-covariant vector consisting of a sequence of SO3part objects, each transforming according
    to a specific irrep of SO(3).
    """

    def __init__(self):
        self.obj=_SO3vecB.zero(1,1,1)

    def __init__(self, x):
        if(isinstance(x,SO3vecB)):
            super().__init__(x)
            self.obj=x.obj
            return
        if(isinstance(x,_SO3vecB)):
            self.obj=x
            return


# ---- Static constructors ------------------------------------------------------------------------------


    @classmethod
    def raw(b, _tau, _dev=0):
        "Construct a zero SO3vec object of given type _tau."
        R = SO3vecB(1)
        r.obj=_SO3vecB.raw(b,_tau,_dev)
        return R

    @classmethod
    def zeros(b, _tau, _dev=0):
        "Construct a zero SO3vec object of given type _tau."
        R = SO3vecB(1)
        r.obj=_SO3vecB.zero(b,_tau,_dev)
        return R

    @classmethod
    def randn(b, _tau, _dev=0):
        "Construct a zero SO3vec object of given type _tau."
        R = SO3vecB(1)
        r.obj=_SO3vecB.gaussian(b,_tau,_dev)
        return R

    @classmethod
    def Fraw(b, _tau, _dev=0):
        "Construct a zero SO3vec object of given type _tau."
        R = SO3vecB(1)
        r.obj=_SO3vecB.Fraw(b,_tau,_dev)
        return R

    @classmethod
    def Fzeros(b, _tau, _dev=0):
        "Construct a zero SO3vec object of given type _tau."
        R = SO3vecB(1)
        r.obj=_SO3vecB.Fzero(b,_tau,_dev)
        return R

    @classmethod
    def Frandn(b, _tau, _dev=0):
        "Construct a zero SO3vec object of given type _tau."
        R = SO3vecB(1)
        r.obj=_SO3vecB.Fgaussian(b,_tau,_dev)
        return R

    @classmethod
    def fromTorch(*args):
        return SO3vecB_InitFromTorchTensorsFn.apply(*args)

    #@classmethod
    #def spharm(self, l, X):
    #    x=_SO3partB.zero(X.size(0),l,X.size(2))
    #    x.add_spharm(X)
    #    return SO3partB(x)

    #@classmethod
    #def zeros_like(self,x):
    #    r=SO3partB([1])
    #    r.obj=_SO3partB.zeros_like(x.obj)
    #    return r

    # ---- Access -------------------------------------------------------------------------------------------


    def get_dev(self):
        return self.obj.get_dev()

    def getb(self):
        return self.obj.getb()

    def get_tau(self):
        return self.obj.get_tau()

    def get_maxl(self):
        return self.obj.get_maxl()

    def _get_grad(self):
        return self.obj.get_grad()
    
    def _view_of_grad(self):
        return self.obj.view_of_grad()
    
    def get_grad(self):
        R=SO3vecB(1)
        R.obj=self.obj.get_grad()
        return R
    
    def view_of_grad(self):
        R=SO3vecB(1)
        R.obj=self.obj.view_of_grad()
        return R
    
    def add_to_grad(self,x):
        R=SO3vecB(1)
        R.obj=self.obj.add_to_grad(x.obj.get_grad())
        return R

    def torch(self):
        return SO3vecB_ToTorchTensorsFn.apply(self)


    # -------------------------------------------------------------------------------------------------------


    def __add__(self,y):
        print("add")
        r=SO3vecB(1)
        r.obj=obj.plus(y)
        return r
        
    def __radd__(self,y):
        print("radd")
        obj.add(y)
        

    # ---- CG-products ---------------------------------------------------------------------------------------


    def CGproduct(self, y, maxl=-1):
        """
        Compute the l component of the Clesbsch--Gordan product of this SO3part with another SO3part y.
        """
        return SO3vecB_CGproductFn.apply(self,y,maxl)

    def DiagCGproduct(self, y, maxl=-1):
        """
        Compute the l component of the diagonal Clesbsch--Gordan product of this SO3part with another SO3part y.
        """
        return SO3vecB_DiagCGproductFn.apply(self,y,maxl)


    def Fproduct(self, y, maxl=-1):
        """
        Compute the l component of the diagonal Clesbsch--Gordan product of this SO3part with another SO3part y.
        """
        return SO3vecB_DiagCGproductFn.apply(self,y,maxl)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ----------------------------------------------------------------------------------------------------------
# ---- Autograd functions -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3vecB_CGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,maxl):
        r=x.CGproduct(y,maxl)
        ctx.x=x
        ctx.y=y
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx, g):
        xg=SO3vecB(1)
        yg=SO3vecB(1)
        ctx.x._view_of_grad().addCGproduct_back0(ctx.r._get_grad(),ctx.y.obj)
        ctx.y._view_of_grad().addCGproduct_back1(ctx.r._get_grad(),ctx.x.obj)
        return xg,yg,None


class SO3vecB_DiagCGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l):
        r=x.DiagCGproduct(y,maxl)
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


class SO3vecB_FproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l):
        r=x.Fproduct(y,maxl)
        ctx.x=x
        ctx.y=y
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx, g):
        xg=SO3partB(1)
        yg=SO3partB(1)
        ctx.x._view_of_grad().addFproduct_back0(ctx.r._get_grad(),ctx.y.obj)
        ctx.y._view_of_grad().addFproduct_back1(ctx.r._get_grad(),ctx.x.obj)
        return xg,yg,None


class SO3vecB_InitFromTorchTensorsFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,*args):
        return SO3vecB(*args)

    @staticmethod
    def backward(ctx,g):
        return g.obj.view_of_grad().torch()


class SO3vecB_ToTorchTensorsFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        return obj.view_of_grad().torch()

    @staticmethod
    def backward(ctx,g):
        return x.add_to_grad(_SO3vecB(g))

    
# ----------------------------------------------------------------------------------------------------------
# ---- Other functions --------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


#def CGproduct(x, y, maxl=-1):
#    return x.CGproduct(y, maxl)


#def DiagCGproduct(x, y, maxl=-1):
#    return x.DiagCGproduct(y, maxl)


