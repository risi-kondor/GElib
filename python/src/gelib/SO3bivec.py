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
from gelib_base import SO3bitype as _SO3bitype
from gelib_base import CGtransform as _CGtransform
from gelib_base import SO3bipart as _SO3bipart
from gelib_base import SO3bivec as _SO3bivec

from gelib import SO3bipart as SO3bipart
from gelib import SO3vecC as SO3vecC


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


class SO3bivec(torch.Tensor):

    @classmethod
    def dummy(self):
        R=SO3bibivec(1)
        #R.obj=_ptensors0.dummy()
        return R

    @classmethod
    def zeros(self,b,_tau,device='cpu'):
        R=SO3bivec(1)
        R.obj=_SO3bivec.zero(b,_SO3bitype(_tau),device_id(device))
        return R

    @classmethod
    def randn(self,b,_tau,device='cpu'):
        R=SO3bivec(1)
        R.obj=_SO3bivec.gaussian(b,_SO3bitype(_tau),device_id(device))
        return R

    @classmethod
    def from_torch(self,T):
        return SO3bivec_fromTorchFn.apply(T)


    # ---- Access ------------------------------------------------------------------------------------------


    def get_dev(self):
        return self.obj.device()

    def getb(self):
        return self.obj.getb()

    def getn(self):
        return self.obj.getn()

    def part(self,l1,l2):
        return SO3bivec_getPartFn.apply(self,l1,l2)


    def get_grad(self):
        R=SO3bivec(1)
        R.obj=self.obj.get_grad()
        return R

    def torch(self):
        return SO3bivec_toTorchFn.apply(self)


    # ---- Products -----------------------------------------------------------------------------------------


    def CGtransform(self, maxl):
        """
        Compute the CGtransform up to maxl.
        """
        return SO3bivec_CGtransformFn.apply(self,maxl)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ----------------------------------------------------------------------------------------------------------
# ---- Autograd functions -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3bivec_fromTorchFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,*args):
        r=SO3bivec(1)
        r.obj=_SO3bivec(*args)
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx,g):
        return tuple(ctx.r.obj.get_grad().torch())


class SO3bivec_toTorchFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        return tuple(x.obj.torch())
 
    @staticmethod
    def backward(ctx,*g):
        ctx.x.obj.add_to_grad(_SO3bivec(*g))
        return SO3bivec.dummy()


class SO3bivec_getPartFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,l1,l2):
        r=SO3bipart(1)
        r.obj=x.obj.part(l1,l2)
        ctx.x=x
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.obj.get_part_back(ctx.r.getl1(),ctx.r.getl2(),ctx.r.obj)
        return SO3bivec.dummy(), None


class SO3bivec_CGtransformFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, maxl):
        tau=_CGtransform(x.obj.get_tau(),maxl)
        r = SO3vecC.zeros(x.getb(),tau,x.get_device())
        x.obj.add_CGtransform_to(r.obj)
        ctx.x=x
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx,g):
        x=ctx.x
        r=ctx.r
        x.obj.add_CGtransform_back(r.obj)
        return SO3bivec.dummy(),None








