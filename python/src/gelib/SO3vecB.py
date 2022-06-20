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
        self.obj=_SO3partB.zero(1,1,1)

    def __init__(self, x):
        if(isinstance(x,SO3vecB)):
            super().__init__(x)
            self.obj=x.obj
            return
        if(isinstance(x,_SO3vecB)):
            self.obj=x
            return


# ---- Static constructors ------------------------------------------------------------------------------


    @staticmethod
    def raw(b, _tau, _dev=0):
        "Construct a zero SO3vec object of given type _tau."
        R = SO3vecB(1)
        r.obj=_SO3vecB.raw(b,_tau,_dev)
        return R

    @staticmethod
    def zeros(b, _tau, _dev=0):
        "Construct a zero SO3vec object of given type _tau."
        R = SO3vecB(1)
        r.obj=_SO3vecB.zero(b,_tau,_dev)
        return R

    @staticmethod
    def randn(b, _tau, _dev=0):
        "Construct a zero SO3vec object of given type _tau."
        R = SO3vecB(1)
        r.obj=_SO3vecB.gaussian(b,_tau,_dev)
        return R

    @staticmethod
    def Fraw(b, _tau, _dev=0):
        "Construct a zero SO3vec object of given type _tau."
        R = SO3vecB(1)
        r.obj=_SO3vecB.Fraw(b,_tau,_dev)
        return R

    @staticmethod
    def Fzeros(b, _tau, _dev=0):
        "Construct a zero SO3vec object of given type _tau."
        R = SO3vecB(1)
        r.obj=_SO3vecB.Fzero(b,_tau,_dev)
        return R

    @staticmethod
    def Frandn(b, _tau, _dev=0):
        "Construct a zero SO3vec object of given type _tau."
        R = SO3vecB(1)
        r.obj=_SO3vecB.Fgaussian(b,_tau,_dev)
        return R

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

    #def torch(self):
    #    return SO3vecB_ToTorchTensorFn.apply(self)


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
        ctx.l=l
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
        ctx.l=l
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
        return SO3partB(x)

    @staticmethod
    def backward(ctx,g):
        return g.obj.view_of_grad().torch()


class SO3partB_ToTorchTensorFn(torch.autograd.Function):

    @staticmethod
    def backward(ctx,x):
        ctx.x=x
        return obj.view_of_grad().torch()

    @staticmethod
    def forward(ctx,g):
        assert(g.dim()==4)
        assert(g.size(3)==2)
        assert(g.size(1)%2==1)
        return x.add_to_grad(_SO3partB(g))

    
# ----------------------------------------------------------------------------------------------------------
# ---- Other functions --------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


#def CGproduct(x, y, maxl=-1):
#    return x.CGproduct(y, maxl)


#def DiagCGproduct(x, y, maxl=-1):
#    return x.DiagCGproduct(y, maxl)


