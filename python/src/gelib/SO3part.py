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


# ----------------------------------------------------------------------------------------------------------
# ---- SO3part ---------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3part(torch.Tensor):
    """
    A collection of vectors that transform according to a specific irreducible representation of SO(3).
    The vectors are stacked into a third order tensor. The first index is the batch index, the second
    is m=-l,...,l, and the third index is the fragment index. 
    """

    def __init__(self, _T):
        self = _T

    # ---- Static constructors -----------------------------------------------------------------------------

    @classmethod
    def zeros(self, b, l, n, _dev=0):
        """
        Create an SO(3)-part consisting of b lots of n vectors transforming according to the l'th irrep of SO(3).
        The vectors are initialized to zero, resulting in an b*(2+l+1)*n dimensional complex tensor of zeros.
        """
        if _dev == 0:
            return SO3part(torch.zeros([b, 2*l+1, n, 2]))
        else:
            return SO3part(torch.zeros([b, 2*l+1, n, 2])).cuda()

    @classmethod
    def randn(self, b, l, n, _dev=0):
        """
        Create an SO(3)-part consisting of b lots of n vectors transforming according to the l'th irrep of SO(3).
        The vectors are initialized as random gaussian vectors, resulting in an b*(2+l+1)*n dimensional random
        complex tensor.
        """
        if _dev == 0:
            return SO3part(torch.randn([b, 2*l+1, n, 2]))
        else:
            return SO3part(torch.randn([b, 2*l+1, n, 2])).cuda()

    @classmethod
    def spharm(self, l, x, y, z, _dev=0):
        """
        Return the spherical harmonics of the vector (x,y,z)
        """
        R = SO3part.zeros(1, l, 1)
        _SO3partB.view(R).add_spharm(x, y, z)
        if _dev > 0:
            return R.cuda()
        return R

    @classmethod
    def spharm(self, l, X, _dev=0):
        """
        Return the spherical harmonics of the vector (x,y,z)
        """
        assert(X.dim()==3)
        R = SO3part.zeros(X.size(0), l, X.size(2))
        _SO3partB.view(R).add_spharm(X)
        if _dev > 0:
            return R.cuda()
        return R

    @classmethod
    def spharmB(self, l, X, _dev=0):
        """
        Return the spherical harmonics of each row of the matrix X.
        """
        R = SO3part.zeros(X.size(0), l, 1)
        _SO3partB.view(R).add_spharmB(X)
        if _dev > 0:
            return R.cuda()
        return R

    @classmethod
    def spharM(self, b, l, n, x, y, z, _dev=0):
        """
        Return the spherical harmonics of the vector (x,y,z)
        """
        R = SO3part.zeros(b, l, n)
        _SO3partB.view(R).add_spharm(x, y, z)
        if _dev > 0:
            return R.cuda()
        return R

    @classmethod
    def Fzeros(self, b, l, _dev=0):
        """
        Create an SO(3)-part corresponding to the l'th matrix in the Fourier transform of a function on SO(3).
        This gives a b*(2+l+1)*(2l+1) dimensional complex tensor. 
        """
        if _dev == 0:
            return SO3part(torch.zeros([b, 2*l+1, 2*l+1, 2]))
        else:
            return SO3part(torch.zeros([b, 2*l+1, 2*l+1, 2])).cuda()

    @classmethod
    def Frandn(self, b, l, _dev=0):
        """
        Create an SO(3)-part corresponding to the l'th matrix in the Fourier transform of a function on SO(3).
        This gives a b*(2+l+1)*(2l+1) dimensional complex random tensor. 
        """
        if _dev == 0:
            return SO3part(torch.randn([b, 2*l+1, 2*l+1, 2]))
        else:
            return SO3part(torch.randn([b, 2*l+1, 2*l+1, 2])).cuda()

    # ---- Access ------------------------------------------------------------------------------------------

    def getb(self):
        return self.size(0)

    def getl(self):
        return (self.size(1)-1)/2

    def getn(self):
        return self.size(2)

    # ---- Operations --------------------------------------------------------------------------------------

    def rotate(self, R):
        A = _SO3partB.view(self).apply(R)
        return SO3part(A.torch())

    def apply(self, R):
        return SO3part(_SO3partB.view(self).apply(R).torch())


    # ---- Products -----------------------------------------------------------------------------------------

    def CGproduct(self, y, l):
        """
        Compute the l component of the Clesbsch--Gordan product of this SO3part with another SO3part y.
        """
        return SO3part_CGproductFn.apply(self,y,l)

    def DiagCGproduct(self, y, l):
        """
        Compute the l component of the diagonal Clesbsch--Gordan product of this SO3part with another SO3part y.
        """
        return SO3part_DiagCGproductFn.apply(self,y,l)


    # ---- I/O ----------------------------------------------------------------------------------------------

    def __repr__(self):
        u=_SO3partB.view(self)
        return u.__repr__()

    def __str__(self):
        u=_SO3partB.view(self)
        return u.__str__()


# ----------------------------------------------------------------------------------------------------------
# ---- Autograd functions -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3part_CGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l):
        ctx.l=l
        ctx.save_for_backward(x,y)

        b = x.size(0)
        dev = int(x.is_cuda)
        r = SO3part.zeros(b,l,x.size(2)*y.size(2),dev)

        _x = _SO3partB.view(x)
        _y = _SO3partB.view(y)
        _r = _SO3partB.view(r)
        _r.addCGproduct(_x,_y)

        return r

    @staticmethod
    def backward(ctx, g):

        x,y = ctx.saved_tensors

        xg=torch.zeros_like(x)
        yg=torch.zeros_like(y)

        _x = _SO3partB.view(x)
        _y = _SO3partB.view(y)

        _g = _SO3partB.view(g)
        _xg = _SO3partB.view(xg)
        _yg = _SO3partB.view(yg)

        _xg.addCGproduct_back0(_g, _y)
        _yg.addCGproduct_back1(_g, _x)

        return xg,yg,None


class SO3part_DiagCGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l):
        ctx.l=l
        assert x.size(2)==y.size(2)
        ctx.save_for_backward(x,y)

        b = x.size(0)
        dev = int(x.is_cuda)
        r = SO3part.zeros(b,l,x.size(2),dev)

        _x = _SO3partB.view(x)
        _y = _SO3partB.view(y)
        _r = _SO3partB.view(r)
        _r.addDiagCGproduct(_x,_y)

        return r

    @staticmethod
    def backward(ctx, g):

        x,y = ctx.saved_tensors

        xg=torch.zeros_like(x)
        yg=torch.zeros_like(y)

        _x = _SO3partB.view(x)
        _y = _SO3partB.view(y)

        _g = _SO3partB.view(g)
        _xg = _SO3partB.view(xg)
        _yg = _SO3partB.view(yg)

        _xg.addDiagCGproduct_back0(_g, _y)
        _yg.addDiagCGproduct_back1(_g, _x)

        return xg,yg,None


# ----------------------------------------------------------------------------------------------------------
# ---- Other functions --------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def CGproduct(x, y, maxl=-1):
    return x.CGproduct(y, maxl)


def DiagCGproduct(x, y, maxl=-1):
    return x.DiagCGproduct(y, maxl)


