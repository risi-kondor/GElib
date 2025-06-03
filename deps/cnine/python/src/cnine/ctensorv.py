#
# This file is part of cnine, a lightweight C++ tensor library. 
#  
# Copyright (c) 2021, Imre Risi Kondor
#
# This source code file is subject to the terms of the noncommercial 
# license distributed with cnine in the file LICENSE.TXT. Commercial 
# use is prohibited. All redistributed versions of this file (in 
# original or modified form) must retain this copyright notice and 
# must be accompanied by a verbatim copy of the license. 
#
#

import torch
from cnine import ctensor 

class ctensorvar(torch.Tensor):

    def __init__(self):
        self.obj=ctensor.zero(1,1,1)

    def __init__(self, x):
        if(isinstance(x,torch.Tensor)):
           self=ctensorvar_InitFromTorchTensorFn.apply(x)


    # ---- Static constructors ------------------------------------------------------------------------------


    @classmethod
    def raw(self, dims, _dev=0):
        R=ctensorvar(1)
        R.obj=ctensor.raw(dims,_dev)
        return R

    @classmethod
    def zeros(self, dims, _dev=0):
        R=ctensorvar(1)
        R.obj=ctensor.zero(dims,_dev)
        return R

    @classmethod
    def randn(self, dims, _dev=0):
        R=ctensorvar(1)
        R.obj=ctensor.gaussian(dims,_dev)
        return R

    @classmethod
    def zeros_like(self,x):
        r=ctensorvar(1)
        r.obj=ctensor.zeros_like(x.obj)
        return r

    @classmethod
    def init(self, x):
        return ctensorvar_InitFromTorchTensorFn.apply(x)


    # -------------------------------------------------------------------------------------------------------


    def get_dev(self):
        return self.obj.get_dev()

    def _get_grad(self):
        return self.obj.get_grad()
    
    def _view_of_grad(self):
        return self.obj.view_of_grad()
    
    def get_grad(self):
        R=ctensorvar(1)
        R.obj=self.obj.get_grad()
        return R
    
    def view_of_grad(self):
        R=ctensorvar(1)
        R.obj=self.obj.view_of_grad()
        return R
    
    def add_to_grad(self,x):
        R=ctensorvar(1)
        R.obj=self.obj.add_to_grad(x.obj.get_grad())
        return R

    def torch(self):
        return ctensorvar_ToTorchTensorFn.apply(self)


    # -------------------------------------------------------------------------------------------------------


    def __add__(self,y):
        r=ctensorvar(1)
        r.obj=obj.plus(y)
        return r
        
    def __radd__(self,y):
        obj.add(y)
        

    # -------------------------------------------------------------------------------------------------------


    def diff2(self,y):
        return ctensorvar_diff2Fn.apply(self,y)

    def inp(self,y):
        return ctensorvar_inpFn.apply(self,y)

    def norm2(self):
        return ctensorvar_norm2Fn.apply(self)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ----------------------------------------------------------------------------------------------------------
# ---- Autograd functions -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class ctensorvar_InitFromTorchTensorFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        r=ctensorvar(1)
        r.obj=ctensor(x)
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx,g):
        return ctx.r.obj.view_of_grad().torch()


class ctensorvar_ToTorchTensorFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        return x.obj.torch()

    @staticmethod
    def backward(ctx,g):
        return ctx.x.obj.add_to_grad(ctensor(g))

    
class ctensorvar_norm2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        return torch.tensor([x.obj.norm2()])

    def backward(ctx,g):
        xg=ctensorvar(1)
        ctx.x._view_of_grad().add_conj(ctx.x.obj,g[0].item()*2.)
        return xg


class ctensorvar_inpFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y):
        ctx.x=x
        ctx.y=y
        return torch.tensor([x.obj.inp(y.obj)])

    def backward(ctx,g):
        xg=ctensorvar(1)
        yg=ctensorvar(1)
        ctx.x._view_of_grad().add_conj(ctx.y.obj,g[0].item())
        ctx.y._view_of_grad().add_conj(ctx.x.obj,g[0].item())
        return xg,yg


class ctensorvar_diff2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y):
        ctx.x=x
        ctx.y=y
        return torch.tensor([x.obj.diff2(y.obj)])

    def backward(ctx,g):
        xg=ctensorvar(1)
        yg=ctensorvar(1)
        v=g[0].item()
        ctx.x._view_of_grad().add_conj(ctx.x.obj,v)
        ctx.x._view_of_grad().add_conj(ctx.y.obj,-v)
        ctx.y._view_of_grad().add_conj(ctx.y.obj,v)
        ctx.y._view_of_grad().add_conj(ctx.x.obj,-v)
        return xg,yg


# ----------------------------------------------------------------------------------------------------------
# ---- Functions -------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
