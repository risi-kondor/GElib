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
from gelib_base import SO3bipart as _SO3bipart
from gelib_base import SO3bivec as _SO3bivec
from gelib_base import SO3bivecArray as _SO3bivecArr

from gelib import SO3bipartArr as SO3bipartArr


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


class SO3bivecArr(torch.Tensor):

    @classmethod
    def dummy(self):
        R=SO3bibivecArr(1)
        #R.obj=_ptensors0.dummy()
        return R

    @classmethod
    def zeros(self,b,adims,_tau,device='cpu'):
        R=SO3bivecArr(1)
        R.obj=_SO3bivecArr.zero(b,adims,_SO3bitype(_tau),device_id(device))
        return R

    @classmethod
    def randn(self,b,adims,_tau,device='cpu'):
        R=SO3bivecArr(1)
        R.obj=_SO3bivecArr.gaussian(b,adims,_SO3bitype(_tau),device_id(device))
        return R

    @classmethod
    def from_torch(self,T):
        return SO3bivecArr_fromTorchFn.apply(T)


    # ---- Access ------------------------------------------------------------------------------------------


    def get_dev(self):
        return self.obj.device()

    def getb(self):
        return self.obj.getb()

    def get_adims(self):
        return self.obj.get_adims()

    def getn(self):
        return self.obj.getn()

    def part(self,l1,l2):
        return SO3bivecArr_getPartFn.apply(self,l1,l2)


    def get_grad(self):
        R=SO3bivecArr(1)
        R.obj=self.obj.get_grad()
        return R

    def torch(self):
        return SO3bivecArr_toTorchFn.apply(self)


    # ---- Products -----------------------------------------------------------------------------------------


    def CGtransform(self, maxl):
        """
        Compute the CGtransform up to maxl.
        """
        return SO3bivecArr_CGtransformFn.apply(self,maxl)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ----------------------------------------------------------------------------------------------------------
# ---- Autograd functions -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3bivecArr_fromTorchFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,*args):
        r=SO3bivecArr(1)
        r.obj=_SO3bivecArr(*args)
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx,g):
        return tuple(ctx.r.obj.get_grad().torch())


class SO3bivecArr_toTorchFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        return tuple(x.obj.torch())
 
    @staticmethod
    def backward(ctx,*g):
        ctx.x.obj.add_to_grad(_SO3bivecArr(*g))
        return SO3bivecArr.dummy()


class SO3bivecArr_getPartFn(torch.autograd.Function):

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
        return SO3bivecArr.dummy(), None


class SO3bivecArr_CGtransformFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, maxl):
        tau=_CGtransform(x.obj.get_tau(),maxl)
        r = SO3vecArrC.zeros(x.getb(),x.get_adims(),tau,x.get_device())
        x.obj.add_CGtransform_to(r.obj)
        ctx.x=x
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx,g):
        x=ctx.x
        r=ctx.r
        x.obj.add_CGtransform_back(r.obj)
        return SO3bivecArr.dummy(),None






