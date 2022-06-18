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

    def __init__(self, _obj):
        self.obj=_obj
     
    @classmethod
    def raw(self, b, l, n, _dev=0):
        return SO3partB(_SO3partB.raw(b,l,n,_dev))

    @classmethod
    def zeros(self, b, l, n, _dev=0):
        r=SO3partB(1)
        r.obj=_SO3partB.zero(b,l,n,_dev)
        return r
        #return SO3partB(_SO3partB.zero(b,l,n,_dev))

    @classmethod
    def randn(self, b, l, n, _dev=0):
        r=SO3partB(1)
        r.obj=_SO3partB.gaussian(b,l,n,_dev)
        return r
        #return SO3partB(_SO3partB.gaussian(b,l,n,_dev))

    @classmethod
    def Fraw(self, b, l, _dev=0):
        return SO3partB(_SO3partB.Fraw(b,l,_dev))

    @classmethod
    def Fzeros(self, b, l, n, _dev=0):
        return SO3partB(_SO3partB.Fzero(b,l,_dev))

    @classmethod
    def Frandn(self, b, l, n, _dev=0):
        return SO3partB(_SO3partB.Fgaussian(b,l,_dev))

    @classmethod
    def spharm(self, l, X):
        x=_SO3partB.zero(X.size(0),l,X.size(2))
        x.add_spharm(X)
        return SO3partB(x)


    # -------------------------------------------------------------------------------------------------------


    def get_dev(self):
        return self.obj.get_dev()

    def getb(self):
        return self.obj.getb()

    def getl(self):
        return self.obj.getl()

    def getn(self):
        return self.obj.getn()


    # ---- CG-products ---------------------------------------------------------------------------------------


    def CGproduct(self, y, l):
        """
        Compute the l component of the Clesbsch--Gordan product of this SO3part with another SO3part y.
        """
        return SO3partB_CGproductFn.apply(self,y,l)

    # ---- I/O ----------------------------------------------------------------------------------------------

    def __repr__(self):
        print("iii")
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ----------------------------------------------------------------------------------------------------------
# ---- Autograd functions -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3partB_CGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l):
        ctx.l=l
        ctx.save_for_backward(x,y)
        r=SO3partB.zeros(x.getb(),l,x.getn()*y.getn(),x.get_dev())
        print(x.obj)
        print(y.obj)
        print(r.obj)
        r.obj.addCGproduct(x.obj,y.obj,l)
        return r

    @staticmethod
    def backward(ctx, g):
        x,y = ctx.saved_tensors
        xg=_SO3partB.zeros_like(x)
        yg=_SO3partB.zeros_like(y)
        xg.obj.addCGproduct_back0(g.obj, y.obj)
        yg.obj.addCGproduct_back1(g.obj, x.obj)
        return xg,yg,None


# ----------------------------------------------------------------------------------------------------------
# ---- Other functions --------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


#def CGproduct(x, y, maxl=-1):
#    return x.CGproduct(y, maxl)


#def DiagCGproduct(x, y, maxl=-1):
#    return x.DiagCGproduct(y, maxl)


