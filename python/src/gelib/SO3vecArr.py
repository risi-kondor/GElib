# This file is part of GElib, a C++/CUDA library for group
# equivariant tensor operations. 
# 
# Copyright (c) 2024, Imre Risi Kondor
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Dict
import torch

import gelib_base as gb
from gelib import *


# ----------------------------------------------------------------------------------------------------------
# ---- SO3vec ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3vecArr:
    """
    An array of SO(3)-covariant vectors. 
    """

    def __init__(self,*args):
        self.parts : Dict[int, SO3partArr]={}
        if args is None:
            return
        
        if len(args) == 1 and \
                not isinstance(args[0], torch.Tensor) and \
                hasattr(args[0], '__iter__'):
            args = list(args[0])
        
        for x in args:
            assert isinstance(x,torch.Tensor)
            p=SO3partArr(x)
            self.parts[p.getl()]=x
            

    # ---- Static constructors ------------------------------------------------------------------------------


    @classmethod
    def zeros(self,b,adims,tau,device='cpu'):
        "Construct a zero SO3vecArr object of given type _tau."
        if adims == None:
            adims = []
        R=SO3vecArr()
        if isinstance(tau,dict):
            for l,n in tau.items():
                R.parts[l]=SO3partArr.zeros(b,adims,l,n,device=device)
        else:
            assert hasattr(tau, '__iter__')
            for i in range(len(tau)):
                R.parts[i] = SO3partArr.zeros(b, adims, i, tau[i], device)
        return R
    
    @classmethod
    def ones(self,b,adims,tau,device='cpu'):
        "Construct a zero SO3vecArr object of given type _tau."
        if adims == None:
            adims = []
        R=SO3vecArr()
        if isinstance(tau,dict):
            for l,n in tau.items():
                R.parts[l]=SO3partArr.ones(b,adims,l,n,device=device)
        else:
            assert hasattr(tau, '__iter__')
            for i in range(len(tau)):
                R.parts[i] = SO3partArr.ones(b, adims, i, tau[i], device)
        return R

    @classmethod
    def randn(self,b,adims,tau,device='cpu'):
        """
        Construct a random SO3vecArr object of given type _tau.
        >>> import gelib
        >>> import torch
         >>> g = torch.manual_seed(0)
        >>> v = gelib.SO3vec.randn(1,[2,3,1])
        >>> v
        <GElib::SO3vecB of type (2,3,1) [b=1]>
        >>> print(v)
        Part l=0:
          [ (1.08965,-0.207486) (-1.54064,0.401942) ]
        <BLANKLINE>
        <BLANKLINE>
        Part l=1:
          [ (-0.251821,0.309163) (1.02082,0.188125) (0.117702,0.618281) ]
          [ (-0.101451,-0.0789197) (-0.433869,0.0223394) (1.41774,0.0379977) ]
          [ (0.437032,-0.291895) (-0.594723,-1.63769) (-0.0723438,0.560342) ]
        <BLANKLINE>
        <BLANKLINE>
        Part l=2:
          [ (-0.151452,-0.305448) (-0.500545,-0.0752602) (-0.878744,-0.336747) (-0.485018,-1.0643) (0.1803,1.2758) ]
        <BLANKLINE>
        <BLANKLINE>
        """
        if adims == None:
            adims = []
        R=SO3vecArr()
        if isinstance(tau,dict):
            for l,n in tau.items():
                R.parts[l]=SO3partArr.randn(b,adims,l,n,device=device)
        else:
            assert hasattr(tau, '__iter__')
            for i in range(len(tau)):
                R.parts[i] = SO3partArr.randn(b, adims, i, tau[i], device)
        return R

    @classmethod
    def Fzeros(self,b,adims,tau,device='cpu'):
        if adims == None:
            adims = []
        R=SO3vecArr()
        if isinstance(tau,dict):
            for l,n in tau.items():
                R.parts[l]=SO3part.Fzeros(b,l,n,device=device)
        return R

    @classmethod
    def Frandn(self,b,adims,tau,device='cpu'):
        if adims == None:
            adims = []
        R=SO3vecArr()
        if isinstance(tau,dict):
            for l,n in tau.items():
                R.parts[l]=SO3partArr.Frandn(b,adims,l,n,device=device)
        return R

    @classmethod
    def spharm(self, l : int, X : torch.Tensor, device : str = 'cpu') -> 'SO3vecArr':
        """
        Return the spherical harmonics of the vector (x,y,z) for all ro <= l.
        """
        R=SO3vecArr()
        R.parts[l]=SO3partArr.spharm(l,X,device=device)
        return R

    @classmethod
    def zeros_like(self, x):
        R=SO3vecArr()
        for l,p in self.tau.parts.items():
            R.parts[l]=SO3partArr.zeros_like(p)
        return R

    @classmethod
    def randn_like(self, x):
        R=SO3vecArr()
        for l,p in x.parts.items():
            R.parts[l]=SO3partArr.randn_like(p)
        return R

    def backend(self):
        return gb.SO3vec.view(self.parts)


    # ---- Access -------------------------------------------------------------------------------------------


    def getb(self):
        return self.parts[min(self.parts)].getb()

    def get_adims(self):
        return self.parts[min(self.parts)].get_adims()

    def tau(self):
        "Return the 'type' of the SO3vec, i.e., how many components it has corresponding to l=0,1,2,..."
        r={}
        for l,p in self.parts.items():
            r[l]=p.getn()
        return r
    
    def get_adims(self):
        return self.parts[min(self.parts)].get_adims()

    def get_type(self):
        "Return the 'type' of the SO3vec, i.e., how many components it has corresponding to l=0,1,2,..."
        r={}
        for l,p in self.parts.items():
            r[l]=p.getn()
        return r

    def l_max(self) -> int:
        if len(self.parts) == 0:
            return 0
        return max(self.parts.keys())

    def requires_grad_(self):
        for l,p in self.parts.items():
            p.requires_grad_()

    def get_grad(self):
        return SO3vecArr(*[p.grad for l,p in self.parts.items()])

    def l_max(self) -> int:
        if len(self.parts) == 0:
            return 0
        return max(self.parts.keys())


    # ---- Operations ---------------------------------------------------------------------------------------


    def apply(self, R):
        "Apply the group element to this vector"
        r = SO3vecArr()
        for l,p in self.parts.items():
            r.parts[l]=p.apply(R)
        return r

    def odot(self,y):
        assert(list(self.parts.keys())==list(y.parts.keys()))
        return sum([self.parts[l].odot(y.parts[l]) for l in self.parts.keys()])

    def __add__(self,y):
        assert(list(self.parts.keys())==list(y.parts.keys()))
        return SO3vecArr(*[self.parts[l]+y.parts[l] for l in self.parts.keys()])

    def gather(self,gmap,dim=0):
        """
        Gather the elements of this SO3vecArr into a new SO3vecArr according to the gather_map
        """
        return SO3vecArr(*[p.gather(gmap,dim) for p in self.parts.values])

        
    # ---- Products -----------------------------------------------------------------------------------------


    def CGproduct(self, y, maxl=-1):
        """
        Compute the full Clesbsch--Gordan product of this SO3vec with another SO3vec y.
        """

        assert isinstance(y, SO3vecArr)
        if maxl == None:
            maxl = -1

        xparts=list(self.parts.values())
        yparts=list(y.parts.values())
        rparts =list(SO3vecArr_CGproductFn.apply(len(xparts), len(yparts), maxl,*(xparts+yparts)))
        return SO3vecArr(*rparts)

    def DiagCGproduct(self, y, maxl=-1):
        """
        Compute the diagonal Clesbsch--Gordan product of this SO3vec with another SO3vec y.
        """
        assert isinstance(y, SO3vecArr)
        if maxl == None:
            maxl = -1

        xparts=list(self.parts.values())
        yparts=list(y.parts.values())
        rparts =list(SO3vecArr_DiagCGproductFn.apply(len(xparts), len(yparts), maxl,*(xparts+yparts)))
        return SO3vecArr(*rparts)


    # ---- I/O ----------------------------------------------------------------------------------------------

    def __repr__(self):
        return self.backend().__repr__()

    def __str__(self):
        return self.backend().__str__()
    
    @property
    def dtype(self) -> torch.dtype:
        if len(self.parts) == 0:
            return torch.cfloat
        key, val = self.parts[0]
        return val.dtype



# ----------------------------------------------------------------------------------------------------------
# ---- Autograd functions -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3vecArr_GatherFn(torch.autograd.Function): # scrap this

    @staticmethod
    def forward(ctx, k1, gmap, d, *args):
        ctx.k1 = k1
        ctx.gmap=gmap
        ctx.save_for_backward(*args)
        ctx.adims=x.get_adims()
        new_adims=ctx.adims
        new_adims[d]=gmap.d_out()
        r=SO3partArr.zeros(x.getb(),ctx.new_adims,x.getl(),x.getn(),device=x.device)
        r.backend().add_gather(x,gmap,d)
        return r

    @staticmethod
    def backward(ctx,rg):
        xg=SO3partArr.zeros(rg.getb(),ctx.adims,rg.getl(),rg.getn(),device=rg.device)
        xg.backend().add_gather_back(rg,gmap,d)
        return xg,None,None


class SO3vecArr_CGproductFn(torch.autograd.Function):

    def __init__(self):
        self.is_sparse=False

    @staticmethod
    def forward(ctx, k1, k2, maxl, *args):
        ctx.k1 = k1
        ctx.k2 = k2
        ctx.save_for_backward(*args)

        x=gb.SO3vec.view(args[0:k1])
        y=gb.SO3vec.view(args[k1:k1+k2])
        b=common_batch(args[0],args[k1])
        adims=args[0].get_adims()
        tau=x.get_tau().CGproduct(y.get_tau())
        rparts=MakeZeroSO3partArrs(b,adims,tau.get_parts(),args[0].device)
        r=gb.SO3vec.view(rparts)
        r.addCGproduct(x,y)

        return tuple(rparts)

    @staticmethod
    def backward(ctx, *args):

        k1 = ctx.k1
        k2 = ctx.k2
        grads=[torch.zeros_like(x) for x in ctx.saved_tensors]

        x=gb.SO3vec.view(ctx.saved_tensors[0:k1])
        y=gb.SO3vec.view(ctx.saved_tensors[k1:k1+k2])
        g=gb.SO3vec.view(args)
        xg=gb.SO3vec.view(grads[0:k1])
        yg=gb.SO3vec.view(grads[k1:k1+k2])
        xg.addCGproduct_back0(g,y)
        yg.addCGproduct_back1(g,x)

        return tuple([None,None,None]+grads)


class SO3vecArr_DiagCGproductFn(torch.autograd.Function):

    def __init__(self):
        self.is_sparse=False

    @staticmethod
    def forward(ctx, k1, k2, maxl, *args):
        ctx.k1 = k1
        ctx.k2 = k2
        ctx.save_for_backward(*args)

        x=gb.SO3vec.view(args[0:k1])
        y=gb.SO3vec.view(args[k1:k1+k2])
        b=common_batch(args[0],args[k1])
        adims=args[0].get_adims()
        tau=x.get_tau().DiagCGproduct(y.get_tau())
        rparts=MakeZeroSO3partArrs(b,adims,tau.get_parts(),args[0].device)
        r=gb.SO3vec.view(rparts)
        r.addDiagCGproduct(x,y)

        return tuple(rparts)

    @staticmethod
    def backward(ctx, *args):

        k1 = ctx.k1
        k2 = ctx.k2
        grads=[torch.zeros_like(x) for x in ctx.saved_tensors]

        x=gb.SO3vec.view(ctx.saved_tensors[0:k1])
        y=gb.SO3vec.view(ctx.saved_tensors[k1:k1+k2])
        g=gb.SO3vec.view(args)
        xg=gb.SO3vec.view(grads[0:k1])
        yg=gb.SO3vec.view(grads[k1:k1+k2])
        xg.addDiagCGproduct_back0(g,y)
        yg.addDiagCGproduct_back1(g,x)

        return tuple([None,None,None]+grads)


class SO3vecArr_FproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, k1, k2, _maxl, *args):
        ctx.k1 = k1
        ctx.k2 = k2
        ctx.save_for_backward(*args)

        b = args[0].size(0)
        if _maxl == -1:
            maxl = k1+k2-2
        else:
            maxl = _maxl

        r = makeZeroSO3FpartArrs(b, maxl, args[0].device)

        _x = _SO3vecB.view(args[0:k1])
        _y = _SO3vecB.view(args[k1:k1+k2])
        _r = _SO3vecB.view(r)
        _r.addFproduct(_x, _y)

        return tuple(r)

    @staticmethod
    def backward(ctx, *args):

        k1 = ctx.k1
        k2 = ctx.k2
        # maxl=ctx.maxl

        inputs = ctx.saved_tensors
        assert len(inputs) == k1+k2, "Wrong number of saved tensors."

        grads = [None, None, None]
        for i in range(k1+k2):
            grads.append(torch.zeros_like(inputs[i]))

        _x = _SO3vecB.view(inputs[0:k1])
        _y = _SO3vecB.view(inputs[k1:k1+k2])

        _g = _SO3vecB.view(args)
        _xg = _SO3vecB.view(grads[3:k1+3])
        _yg = _SO3vecB.view(grads[k1+3:k1+k2+3])

        _xg.addFproduct_back0(_g, _y)
        _yg.addFproduct_back1(_g, _x)

        return tuple(grads)


class SO3vecArr_FmodsqFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, k1, _maxl, *args):
        ctx.k1 = k1
        # ctx.k2=k1
        ctx.save_for_backward(*args)

        b = args[0].size(0)
        if _maxl == -1:
            maxl = k1+k1-2
        else:
            maxl = _maxl

        r = makeZeroSO3FpartArrs(b, maxl, args[0].device)

        _x = _SO3vecB.view(args[0:k1])
        _r = _SO3vecB.view(r)
        _r.addFproduct(_x, _x)

        return tuple(r)

    @staticmethod
    def backward(ctx, *args):

        k1 = ctx.k1
        # k2=ctx.k2

        inputs = ctx.saved_tensors
        assert len(inputs) == k1, "Wrong number of saved tensors."

        grads = [None, None]
        for i in range(k1):
            grads.append(torch.zeros_like(inputs[i]))

        _x = _SO3vecB.view(inputs[0:k1])

        _g = _SO3vecB.view(args)
        _xg = _SO3vecB.view(grads[2:k1+2])

        _xg.addFproduct_back0(_g, _x)
        _xg.addFproduct_back1(_g, _x)

        return tuple(grads)


class SO3vec_iFFTFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, N, *args):

        _v=_SO3vecB.view(args)
        b=_v.getb()
        #maxl=_v.get_maxl()
        ctx.save_for_backward(*args)
        
        r=torch.zeros([b,2*N,N,2*N,2],device=args[0].device)
        _r=ctensorb.view(r)
        _v.add_iFFT_to(_r)

        return r

    @staticmethod
    def backward(ctx, fg):

        inputs = ctx.saved_tensors
        grads = [None]
        for inp in inputs:
            grads.append(torch.zeros_like(inp))

        _fg = ctensorb.view(fg)
        _vg=_SO3vecB.view(grads[1:])
        _vg.add_FFT(_fg)
        
        return tuple(grads)


class SO3vec_FFTFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, maxl, f):

        ctx.save_for_backward(f)
        _f = ctensorb.view(f)

        v = makeZeroSO3Fparts(_f.get_dim(0), maxl, _f.get_dev())
        _v=_SO3vecB.view(v)
        _v.add_FFT(_f)

        return tuple(v)

    @staticmethod
    def backward(ctx, vg):

        inputs = ctx.saved_tensors

        fg=torch.zeros_like(inputs[0])
        _fg=ctensorb.view(fg)
        _vg.add_iFFT_to(_fg)
        
        return tuple([None, fg])



# ----------------------------------------------------------------------------------------------------------
# ---- Other functions --------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def CGproduct(x, y, maxl=-1):
    return x.CGproduct(y, maxl)


def DiagCGproduct(x, y, maxl=-1):
    return x.DiagCGproduct(y, maxl)


def Fproduct(x, y, maxl=-1):
    return x.Fproduct(y, maxl)


def Fmodsq(x, a=-1):
    return x.Fmodsq(a)


# ---- Helpers -----------------------------------------------------------------------------------------------




def MakeZeroSO3partArrs(b,adims,tau,device):
    R=[]
    for l,n in tau.items():
        R.append(SO3partArr.zeros(b,adims,l,n,device))
    return R


def makeZeroSO3FpartArrs(b,adims,maxl,device):
    R = []
    for l,n in tau.items():
        R.append(SO3partArr.Fzeros(b,adims,l,device))
    return R


def SO3FFT(f,maxl):
    r=SO3vec()
    r.parts=list(SO3vec_FFTFn.apply(maxl,f))
    return r


def SO3iFFT(v,N):
    return v.iFFT(N)

