# This file is part of GElib, a C++/CUDA library for group
# equivariant tensor operations. 
# 
# Copyright (c) 2023, Imre Risi Kondor
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch
from gelib_base import SO3type as _SO3type
from gelib_base import SO3part as _SO3part
from gelib_base import SO3vec as _SO3vec
from gelib_base import CGproduct as _CGproduct

from gelib import SO3partC as SO3partC


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
# ---- SO3vec ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3vecC(torch.Tensor):

    @classmethod
    def dummy(self):
        R=SO3vecC(1)
        #R.obj=_ptensors0.dummy()
        return R

    @classmethod
    def zeros(self,b,_tau,device='cpu'):
        R=SO3vecC(1)
        R.obj=_SO3vec.zero(b,_SO3type(_tau),device_id(device))
        return R

    @classmethod
    def randn(self,b,_tau,device='cpu'):
        R=SO3vecC(1)
        R.obj=_SO3vec.gaussian(b,_SO3type(_tau),device_id(device))
        return R

    @classmethod
    def from_torch(self,T):
        return SO3vecC_fromTorchFn.apply(T)


    # ---- Access ------------------------------------------------------------------------------------------


    def get_dev(self):
        return self.obj.device()

    def getb(self):
        return self.obj.getb()

    def getl(self):
        return self.obj.getl()

    def getn(self):
        return self.obj.getn()

    def __getitem__(self,i):
        return SO3vecC_getPartFn.apply(self,i)


    def get_grad(self):
        R=SO3vecC(1)
        R.obj=self.obj.get_grad()
        return R

    def torch(self):
        return SO3vecC_toTorchFn.apply(self)


    # ---- Products -----------------------------------------------------------------------------------------


    def CGproduct(self,y,maxl):
        """
        Compute the full Clesbsch--Gordan product of this SO3vec with another SO3vec y.
        """
        return SO3vecC_CGproductFn.apply(self,y,maxl)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ----------------------------------------------------------------------------------------------------------
# ---- Autograd functions -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3vecC_fromTorchFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,*args):
        r=SO3vecC(1)
        r.obj=_SO3vec(*args)
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx,g):
        return tuple(ctx.r.obj.get_grad().torch())


class SO3vecC_toTorchFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        return tuple(x.obj.torch())
 
    @staticmethod
    def backward(ctx,*g):
        ctx.x.obj.add_to_grad(_SO3vec(*g))
        return SO3vecC.dummy()


class SO3vecC_getPartFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,l):
        r=SO3partC(1)
        r.obj=x.obj.part(l)
        ctx.x=x
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.obj.get_part_back(ctx.r.getl(),ctx.r.obj)
        return SO3vecC.dummy(), None


class SO3vecC_CGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,maxl):
        tau=_CGproduct(x.obj.get_tau(),y.obj.get_tau(),maxl)
        r = SO3vecC.zeros(x.getb(),tau,x.get_device())
        r.obj.add_CGproduct(x.obj,y.obj)
        ctx.x=x
        ctx.y=y
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.obj.add_CGtransform_back0(ctx.r.obj)
        ctx.y.obj.add_CGtransform_back1(ctx.r.obj)
        return SO3vecC.dummy(),None






