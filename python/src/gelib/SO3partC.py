# This file is part of GElib, a C++/CUDA library for group
# equivariant tensor operations. 
# 
# Copyright (c) 2022, Imre Risi Kondor
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch

from gelib_base import SO3part as _SO3part


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
# ---- SO3part ---------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------



class SO3partC(torch.Tensor):

    @classmethod
    def dummy(self):
        R=SO3partC(1)
        #R.obj=_ptensors0.dummy()
        return R

    @classmethod
    def zeros(self, b, l, n, device='cpu'):
        R=SO3partC(1)
        R.obj=_SO3part.zero(b,l,n,device_id(device))
        return R

    @classmethod
    def randn(self, b, l, n, device='cpu'):
        R=SO3partC(1)
        R.obj=_SO3part.gaussian(b,l,n,device_id(device))
        return R

    @classmethod
    def from_torch(self,T):
        return SO3partC_fromTorchFn.apply(T)

            
    # ---- Access ------------------------------------------------------------------------------------------


    def get_dev(self):
        return self.obj.device()

    def getb(self):
        return self.obj.getb()

    def getl(self):
        return self.obj.getl()

    def getn(self):
        return self.obj.getn()

    def get_grad(self):
        R=SO3partC(1)
        R.obj=self.obj.get_grad()
        return R

    def torch(self):
        return SO3partC_toTorchFn.apply(self)

    
    # ---- Products -----------------------------------------------------------------------------------------


    def CGproduct(self, y, l):
        """
        Compute the l component of the Clesbsch--Gordan product of this SO3part with another SO3part y.
        """
        return SO3partC_CGproductFn.apply(self,y,l)

    def DiagCGproduct(self, y, l):
        """
        Compute the l component of the diagonal Clesbsch--Gordan product of this SO3part with another SO3part y.
        """
        return SO3partC_DiagCGproductFn.apply(self,y,l)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ----------------------------------------------------------------------------------------------------------
# ---- Autograd functions -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3partC_fromTorchFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        r=SO3partC(1)
        r.obj=_SO3part(x)
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx,g):
        return ctx.r.obj.get_grad().torch()


class SO3partC_toTorchFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        return x.obj.torch()
 
    @staticmethod
    def backward(ctx,g):
        ctx.x.obj.add_to_grad(_SO3part(g))
        return SO3partC.dummy()
    

class SO3partC_CGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l):
        r=SO3partC.zeros(x.getb(),l,x.getn()*y.getn(),x.get_dev())
        r.obj.add_CGproduct(x.obj,y.obj,l)
        #ctx.save_for_backward(r,x,y) doesn't work
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
        return SO3partC.dummy(),SO3partC.dummy(),None


class SO3partC_DiagCGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l):
        r=SO3partC.zeros(x.getb(),l,x.getn(),x.get_dev())
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
        return SO3partC.dummy(),SO3partC.dummy(),None


# ----------------------------------------------------------------------------------------------------------
# ---- Other functions --------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


#def CGproduct(x, y, maxl=-1):
#    return x.CGproduct(y, maxl)


#def DiagCGproduct(x, y, maxl=-1):
#    return x.DiagCGproduct(y, maxl)

