# This file is part of GElib, a C++/CUDA library for group
# equivariant tensor operations. 
# 
# Copyright (c) 2022, Imre Risi Kondor
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch

from gelib_base import SO3partArray as _SO3partArray

from gelib import SO3partC


def device_id(device):
    if device==0:
        return 0
    if device==1:
        return 1
    if device=='cpu':
        return 0
    if device=='gpu':
        return 1
    if device=='cuda':
        return 1
    if device=='cuda:0':
        return 1
    return 0


# ----------------------------------------------------------------------------------------------------------
# ---- SO3partArr ------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------



class SO3partArrC(torch.Tensor):

    @classmethod
    def dummy(self):
        R=SO3partArrC(1)
        #R.obj=_ptensors0.dummy()
        return R

    @classmethod
    def zeros(self, b, adims, l, n, device='cpu'):
        R=SO3partArrC(1)
        R.obj=_SO3partArray.zero(b,adims,l,n,device_id(device))
        return R

    @classmethod
    def randn(self, b, adims, l, n, device='cpu'):
        R=SO3partArrC(1)
        R.obj=_SO3partArray.gaussian(b,adims,l,n,device_id(device))
        return R

    @classmethod
    def from_torch(self,T):
        return SO3partArrC_fromTorchFn.apply(T)

            
    # ---- Access ------------------------------------------------------------------------------------------


    def get_dev(self):
        return self.obj.device()

    def getb(self):
        return self.obj.getb()

    def get_adims(self):
        return self.obj.get_adims()

    def getl(self):
        return self.obj.getl()

    def getn(self):
        return self.obj.getn()

    def cell(self,i):
        return SO3partArrC_getCellFn.apply(self,i)


    def get_grad(self):
        R=SO3partArrC(1)
        R.obj=self.obj.get_grad()
        return R

    def torch(self):
        return SO3partArrC_toTorchFn.apply(self)

    
    # ---- Products -----------------------------------------------------------------------------------------


    def CGproduct(self, y, l):
        """
        Compute the l component of the Clesbsch--Gordan product of this SO3partArr with another SO3partArr y.
        """
        return SO3partArrC_CGproductFn.apply(self,y,l)


    def DiagCGproduct(self, y, l):
        """
        Compute the l component of the diagonal Clesbsch--Gordan product of this SO3partArr with another SO3partArr y.
        """
        return SO3partArrC_DiagCGproductFn.apply(self,y,l)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ----------------------------------------------------------------------------------------------------------
# ---- Autograd functions -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3partArrC_fromTorchFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        r=SO3partArrC(1)
        r.obj=_SO3partArray(x)
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx,g):
        return ctx.r.obj.get_grad().torch()


class SO3partArrC_toTorchFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        return x.obj.torch()
 
    @staticmethod
    def backward(ctx,g):
        ctx.x.obj.add_to_grad(_SO3partArray(g))
        return SO3partArrC.dummy()
    

class SO3partArrC_getCellFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,ix):
        r=SO3partC(1)
        r.obj=x.obj.cell(ix)
        ctx.x=x
        ctx.r=r
        ctx.ix=ix
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.obj.get_cell_back(ctx.ix,ctx.r.obj)
        return SO3partArrC.dummy(), None


class SO3partArrC_CGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l):
        r=SO3partArrC.zeros(x.getb(),x.get_adims(),l,x.getn()*y.getn(),x.get_dev())
        r.obj.add_CGproduct(x.obj,y.obj,l)
        ctx.x=x
        ctx.y=y
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx, g):
        x=ctx.x
        y=ctx.y
        r=ctx.r
        x.obj.add_CGproduct_back0(r.obj,y.obj)
        y.obj.add_CGproduct_back1(r.obj,x.obj)
        return SO3partArrC.dummy(),SO3partArrC.dummy(),None


class SO3partArrC_DiagCGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l):
        r=SO3partArrC.zeros(x.getb(),x.get_adims(),l,x.getn(),x.get_dev())
        r.obj.add_DiagCGproduct(x.obj,y.obj,l)
        ctx.x=x
        ctx.y=y
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx, g):
        x=ctx.x
        y=ctx.y
        r=ctx.r
        x.obj.add_DiagCGproduct_back0(r.obj,y.obj)
        y.obj.add_DiagCGproduct_back1(r.obj,x.obj)
        return SO3partArrC.dummy(),SO3partArrC.dummy(),None


# ----------------------------------------------------------------------------------------------------------
# ---- Other functions --------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


#def CGproduct(x, y, maxl=-1):
#    return x.CGproduct(y, maxl)


#def DiagCGproduct(x, y, maxl=-1):
#    return x.DiagCGproduct(y, maxl)

