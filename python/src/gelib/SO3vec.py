
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
from gelib_base import SO3vecB as _SO3vecB
#from gelib_base import SO3Fvec as _SO3Fvec

from gelib import *

#from SO3part import SO3part


# ----------------------------------------------------------------------------------------------------------
# ---- SO3vec ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3vec:
    """
    An SO(3)-covariant vector consisting of a sequence of SO3part objects, each transforming according
    to a specific irrep of SO(3).
    """

    def __init__(self):
        self.parts = []

    def __init__(self,parts=None):
        self.parts=[]
        if parts is not None:
            for l in range(len(parts)):
                self.parts.append(parts[l])


    # ---- Static constructors ------------------------------------------------------------------------------

    @staticmethod
    def zeros(b, _tau, _dev=0):
        "Construct a zero SO3vec object of given type _tau."
        R = SO3vec()
        for l in range(0, len(_tau)):
            R.parts.append(SO3part.zeros(b, l, _tau[l], _dev))
        return R

    @staticmethod
    def randn(b, _tau, _dev=0):
        "Construct a random SO3vec object of given type _tau."
        R = SO3vec()
        for l in range(0, len(_tau)):
            R.parts.append(SO3part.randn(b, l, _tau[l], _dev))
        return R

    @staticmethod
    def spharm(b, _tau, x, y, z, _dev=0):
        """
        Compute a vector of spherical harmonic coefficients. 
        The values will be duplicated along the batch and channel dimensions.
        """
        R = SO3vec()
        for l in range(0, len(_tau)):
            R.parts.append(SO3part.spharM(b, l, _tau[l], x, y, z, _dev))
        return R

    @staticmethod
    def Fzeros(b, maxl, _dev=0):
        "Construct an SO3vec corresponding the to the Forier matrices 0,1,...maxl of b functions on SO(3)."
        R = SO3vec()
        for l in range(0, maxl+1):
            R.parts.append(SO3part.Fzeros(b, l, _dev))
        return R

    @staticmethod
    def Frandn(b, maxl, _dev=0):
        "Construct a zero SO3Fvec object  with l ranging from 0 to maxl."
        R = SO3vec()
        for l in range(0, maxl+1):
            R.parts.append(SO3part.Frandn(b, l, _dev))
        return R

    @staticmethod
    def zeros_like(x):
        R = SO3vec()
        # b=x.parts[0].dim(0)
        for l in range(0, len(x.parts)):
            R.parts.append(SO3part(torch.zeros_like(x.parts[l])))
        return R

    # ---- Access -------------------------------------------------------------------------------------------

    def getb(self):
        return parts[0].size(0)

    def tau(self):
        "Return the 'type' of the SO3vec, i.e., how many components it has corresponding to l=0,1,2,..."
        r = []
        for l in range(0, len(self.parts)):
            r.append(self.parts[l].size(2))
            # r.append(self.parts[l].getn())
        return r

    # ---- Transport ---------------------------------------------------------------------------------------

    def to(self, device):
        r = SO3vec()
        for p in self.parts:
            r.parts.append(p.to(device))
        return r

    # ---- Operations ---------------------------------------------------------------------------------------

    def rotate(self, R):
        "Apply the group element to this vector"
        r = SO3vec()
        for l in range(0, len(self.parts)):
            r.parts.append(self.parts[l].rotate(R))
        return r

    # ---- Products -----------------------------------------------------------------------------------------

    def CGproduct(self, y, maxl=-1):
        """
        Compute the full Clesbsch--Gordan product of this SO3vec with another SO3vec y.
        """
        r = SO3vec()
        r.parts = list(SO3vec_CGproductFn.apply(len(self.parts), len(y.parts), maxl, *(self.parts+y.parts)))
        return r

    def DiagCGproduct(self, y, maxl=-1):
        """
        Compute the diagonal Clesbsch--Gordan product of this SO3vec with another SO3vec y.
        """
        r = SO3vec()
        r.parts = list(SO3vec_DiagCGproductFn.apply(len(self.parts), len(y.parts), maxl, *(self.parts+y.parts)))
        return r

    def Fproduct(self, y, maxl=-1):
        """
        Compute the Fourier space product of this SO3Fvec with another SO3Fvec y.
        """
        r = SO3vec()
        r.parts = list(SO3vec_FproductFn.apply(len(self.parts), len(y.parts), maxl, *(self.parts+y.parts)))
        return r

    def Fmodsq(self, maxl=-1):
        """
        Compute the Fourier transform of the squared modulus of f.
        """
        r = SO3vec()
        r.parts = list(SO3vec_FmodsqFn.apply(len(self.parts), maxl, *(self.parts)))
        return r

    # ---- Fourier transforms -------------------------------------------------------------------------------

    def iFFT(self,_N):
        return SO3vec_iFFTFn.apply(_N,*(self.parts))
    
    # ---- I/O ----------------------------------------------------------------------------------------------

    def __repr__(self):
        u=_SO3vecB.view(self.parts)
        return u.__repr__()

    def __str__(self):
        u = _SO3vecB.view(self.parts)
        return u.__str__()


# ----------------------------------------------------------------------------------------------------------
# ---- Autograd functions -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3vec_CGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, k1, k2, maxl, *args):
        ctx.k1 = k1
        ctx.k2 = k2
        # ctx.maxl=maxl
        ctx.save_for_backward(*args)

        b = args[0].size(0)
        tau = CGproductType(tau_type(args[0:k1]), tau_type(args[k1:k1+k2]), maxl)
        dev = int(args[0].is_cuda)
        r = MakeZeroSO3parts(b, tau, dev)

        _x = _SO3vecB.view(args[0:k1])
        _y = _SO3vecB.view(args[k1:k1+k2])
        _r = _SO3vecB.view(r)
        _r.addCGproduct(_x, _y)

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

        _xg.addCGproduct_back0(_g, _y)
        _yg.addCGproduct_back1(_g, _x)

        return tuple(grads)


class SO3vec_DiagCGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, k1, k2, maxl, *args):
        ctx.k1 = k1
        ctx.k2 = k2
        # ctx.maxl=maxl
        ctx.save_for_backward(*args)

        b = args[0].size(0)
        tau = DiagCGproductType(tau_type(args[0:k1]), tau_type(args[k1:k1+k2]), maxl)
        dev = int(args[0].is_cuda)
        r = MakeZeroSO3parts(b, tau, dev)

        _x = _SO3vecB.view(args[0:k1])
        _y = _SO3vecB.view(args[k1:k1+k2])
        _r = _SO3vecB.view(r)
        _r.addDiagCGproduct(_x, _y)

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

        _xg.addDiagCGproduct_back0(_g, _y)
        _yg.addDiagCGproduct_back1(_g, _x)

        return tuple(grads)


class SO3vec_FproductFn(torch.autograd.Function):

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
        dev = int(args[0].is_cuda)

        r = makeZeroSO3Fparts(b, maxl, dev)

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


class SO3vec_FmodsqFn(torch.autograd.Function):

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
        dev = int(args[0].is_cuda)

        r = makeZeroSO3Fparts(b, maxl, dev)

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


def Fproduct(x, y, a=-1):
    return x.Fproduct(y, a)


def Fmodsq(x, a=-1):
    return x.Fmodsq(a)


# ---- Helpers -----------------------------------------------------------------------------------------------


def tau_type(x):
    r = []
    for t in x:
        r.append(t.size(2))
    return r


def CGproductType(x, y, maxl=-1):
    if maxl == -1:
        maxl = len(x)+len(y)-2
    maxl = min(maxl, len(x)+len(y)-2)
    r = [0]*(maxl+1)
    for l1 in range(0, len(x)):
        for l2 in range(0, len(y)):
            for l in range(abs(l1-l2), min(l1+l2, maxl)+1):
                r[l] += x[l1]*y[l2]
    return r


def DiagCGproductType(x, y, maxl=-1):
    if maxl == -1:
        maxl = len(x)+len(y)-2
    maxl = min(maxl, len(x)+len(y)-2)
    r = [0]*(maxl+1)
    for l1 in range(0, len(x)):
        for l2 in range(0, len(y)):
            for l in range(abs(l1-l2), min(l1+l2, maxl)+1):
                r[l] += x[l1]
    return r


def MakeZeroSO3parts(b, tau, _dev=0):
    R = []
    for l in range(0, len(tau)):
        R.append(SO3part.zeros(b, l, tau[l], _dev))
    return R


def makeZeroSO3Fparts(b, maxl, _dev=0):
    R = []
    for l in range(0, maxl+1):
        R.append(SO3part.Fzeros(b, l, _dev))
    return R


def SO3FFT(f,maxl):
    r=SO3vec()
    r.parts=list(SO3vec_FFTFn.apply(maxl,f))
    return r


def SO3iFFT(v,N):
    return v.iFFT(N)

