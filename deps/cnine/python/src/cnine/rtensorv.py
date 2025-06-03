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
from cnine import rtensor 

class rtensorvar(torch.Tensor):

    def __init__(self):
        self.obj=rtensor.zero(1,1,1)

    def __init__(self, x):
        if(isinstance(x,torch.Tensor)):
            r=rtensorvar_InitFromTorchTensorFn.apply(x)
            self.obj=r.obj
            

    # ---- Static constructors ------------------------------------------------------------------------------


    @classmethod
    def raw(self, dims, _dev=0):
        R=rtensorvar(1)
        R.obj=rtensor.raw(dims,_dev)
        return R

    @classmethod
    def zeros(self, dims, _dev=0):
        R=rtensorvar(1)
        R.obj=rtensor.zero(dims,_dev)
        return R

    @classmethod
    def randn(self, dims, _dev=0):
        R=rtensorvar(1)
        R.obj=rtensor.gaussian(dims,_dev)
        return R

    @classmethod
    def zeros_like(self,x):
        r=rtensorvar(1)
        r.obj=rtensor.zeros_like(x.obj)
        return r

    @classmethod
    def init(self, x):
        return rtensorvar_InitFromTorchTensorFn.apply(x)


    # -------------------------------------------------------------------------------------------------------


    def get_dev(self):
        return self.obj.get_dev()

    def _get_grad(self):
        return self.obj.get_grad()
    
    def _view_of_grad(self):
        return self.obj.view_of_grad()
    
    def get_grad(self):
        R=rtensorvar(1)
        R.obj=self.obj.get_grad()
        return R
    
    def view_of_grad(self):
        R=rtensorvar(1)
        R.obj=self.obj.view_of_grad()
        return R
    
    def add_to_grad(self,x):
        R=rtensorvar(1)
        R.obj=self.obj.add_to_grad(x.obj.get_grad())
        return R

    def torch(self):
        return rtensorvar_ToTorchTensorFn.apply(self)


    # -------------------------------------------------------------------------------------------------------


    def __add__(self,y):
        r=rtensorvar(1)
        r.obj=obj.plus(y)
        return r
        
    def __radd__(self,y):
        obj.add(y)
        

    # -------------------------------------------------------------------------------------------------------


    def diff2(self,y):
        return rtensorvar_diff2Fn.apply(self,y)

    def inp(self,y):
        return rtensorvar_inpFn.apply(self,y)

    def norm2(self):
        return rtensorvar_norm2Fn.apply(self)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ----------------------------------------------------------------------------------------------------------
# ---- Autograd functions -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class rtensorvar_InitFromTorchTensorFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        r=rtensorvar(1)
        r.obj=rtensor(x)
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx,g):
        return ctx.r.obj.view_of_grad().torch()


class rtensorvar_ToTorchTensorFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        return x.obj.torch()

    @staticmethod
    def backward(ctx,g):
        return ctx.x.obj.add_to_grad(rtensor(g))

    
class rtensorvar_norm2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        return torch.tensor([x.obj.norm2()])

    def backward(ctx,g):
        xg=rtensorvar(1)
        ctx.x._view_of_grad().add(ctx.x.obj,g[0].item()*2.)
        return xg


class rtensorvar_inpFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y):
        ctx.x=x
        ctx.y=y
        return torch.tensor([x.obj.inp(y.obj)])

    def backward(ctx,g):
        xg=rtensorvar(1)
        yg=rtensorvar(1)
        ctx.x._view_of_grad().add(ctx.y.obj,g[0].item())
        ctx.y._view_of_grad().add(ctx.x.obj,g[0].item())
        return xg,yg


class rtensorvar_diff2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y):
        ctx.x=x
        ctx.y=y
        return torch.tensor([x.obj.diff2(y.obj)])

    def backward(ctx,g):
        xg=rtensorvar(1)
        yg=rtensorvar(1)
        v=g[0].item()
        ctx.x._view_of_grad().add(ctx.x.obj,v)
        ctx.x._view_of_grad().add(ctx.y.obj,-v)
        ctx.y._view_of_grad().add(ctx.y.obj,v)
        ctx.y._view_of_grad().add(ctx.x.obj,-v)
        return xg,yg


# ----------------------------------------------------------------------------------------------------------
# ---- Functions -------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def norm2(x):
    return x.norm2()

def inp(x,y):
    return x.inp(y)

def diff2(x,y):
    return x.diff2(y)
