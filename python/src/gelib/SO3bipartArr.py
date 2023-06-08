# This file is part of GElib, a C++/CUDA library for group
# equivariant tensor operations. 
# 
# Copyright (c) 2022, Imre Risi Kondor
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch
#from cnine import ctensorb 
from gelib_base import SO3bipartArray as _SO3bipartArr

from gelib import SO3partArrC as SO3partArr


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



class SO3bipartArr(torch.Tensor):

    @classmethod
    def dummy(self):
        R=SO3bipartArr(1)
        #R.obj=_ptensors0.dummy()
        return R

    @classmethod
    def zeros(self, b, adims, l1, l2, n, device='cpu'):
        R=SO3bipartArr(1)
        R.obj=_SO3bipartArr.zero(b,adims,l1,l2,n,device_id(device))
        return R

    @classmethod
    def randn(self, b, adims, l1, l2, n, device='cpu'):
        R=SO3bipartArr(1)
        R.obj=_SO3bipartArr.gaussian(b,adims,l1,l2,n,device_id(device))
        return R

    @classmethod
    def from_torch(self,T):
        return SO3bipartArr_fromTorchFn.apply(T)

            
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

    def get_grad(self):
        R=SO3bipartArr(1)
        R.obj=self.obj.get_grad()
        return R

    def torch(self):
        return SO3bipartArr_toTorchFn.apply(self)

    
    # ---- Products -----------------------------------------------------------------------------------------


    def CGtransform(self, l):
        """
        Compute the l component of the CGtransform.
        """
        return SO3bipartArr_CGtransformFn.apply(self,l)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ----------------------------------------------------------------------------------------------------------
# ---- Autograd functions -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3bipartArr_fromTorchFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        r=SO3bipartArr(1)
        r.obj=_SO3bipartArr(x)
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx,g):
        return ctx.r.obj.get_grad().torch()


class SO3bipartArr_toTorchFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        return x.obj.torch()
 
    @staticmethod
    def backward(ctx,g):
        ctx.x.obj.add_to_grad(_SO3bipartArr(g))
        return SO3bipartArr.dummy()
    

class SO3bipartArr_CGtransformFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,l):
        print(x.get_dev())
        r=SO3partArr.zeros(x.getb(),x.get_adims(),l,x.getn(),x.get_dev())
        x.obj.add_CGtransform_to(r.obj,0)
        ctx.x=x
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx, g):
        x=ctx.x
        r=ctx.r
        x.obj.add_CGtransform_back(r.obj)
        return SO3bipartArr.dummy(),None


# ----------------------------------------------------------------------------------------------------------
# ---- Other functions --------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def CGtransform(x,l):
    return x.CGtransform(l)
