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
from gelib_base import SO3weights as _SO3weights
#from gelib_base import SO3Fvec as _SO3Fvec

from gelib import *

#from SO3part import SO3part


# ----------------------------------------------------------------------------------------------------------
# ---- SO3vec ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3weights:
    """
    Vector of weight tensors to multiply SO3vec by.
    """

    def __init__(self):
        self.parts = []

    def __init__(self,parts=None):
        self.parts=[]
        if parts is not None:
            for l in range(len(parts)):
                self.parts.append(parts[l])


    # ---- Static constructors ------------------------------------------------------------------------------

    @classmethod
    def zeros(self, _tau1, _tau2, device='cpu'):
        R = SO3weights()
        assert len(_tau1)==len(_tau2)
        for l in range(0, len(_tau1)):
            R.parts.append(torch.zeros([_tau1[l],_tau2[l]],dtype=torch.cfloat,device=device))
        return R

    @classmethod
    def randn(self, _tau1, _tau2, device='cpu'):
        R = SO3weights()
        assert len(_tau1)==len(_tau2)
        for l in range(0, len(_tau1)):
            R.parts.append(torch.randn([_tau1[l],_tau2[l]],dtype=torch.cfloat,device=device))
        return R

    @classmethod
    def Fzeros(self, _tau1, _tau2, device='cpu'):
        R = SO3weights()
        assert len(_tau1)==len(_tau2)
        for l in range(0, len(_tau1)):
            R.parts.append(torch.zeros([2*l+1,2*l+1],dtype=torch.cfloat,device=device))
        return R

    @classmethod
    def Frandn(self, _tau1, _tau2, device='cpu'):
        R = SO3weights()
        assert len(_tau1)==len(_tau2)
        for l in range(0, len(_tau1)):
            R.parts.append(torch.randn([2*l+1,2*l+1],dtype=torch.cfloat,device=device))
        return R

    @classmethod
    def zeros_like(self, x):
        R = SO3weights()
        for l in range(0, len(x.parts)):
            R.parts.append(torch.zeros_like(x.parts[l]))
        return R

    
    # ---- Arithmetic ---------------------------------------------------------------------------------------


    def __mul__(self, w):
        if(isinstance(w,torch.Tensor)):
            R=SO3weights()
            for l in range(len(self.parts)):
                R.parts.append(self.parts[l]*w)
            return R
        raise TypeError("SO3weights can only be multiplied by a scalar tensor.")


    # ---- Access -------------------------------------------------------------------------------------------


    def tau1(self):
        "Return the 'type' of the SO3vec, i.e., how many components it has corresponding to l=0,1,2,..."
        r = []
        for l in range(0, len(self.parts)):
            r.append(self.parts[l].size(0))
        return r

    def requires_grad_(self):
        for p in self.parts:
            p.requires_grad_()
            

    # ---- Transport ---------------------------------------------------------------------------------------

    def to(self, device):
        r = SO3weights()
        for p in self.parts:
            r.parts.append(p.to(device))
        return r


    # ---- I/O ----------------------------------------------------------------------------------------------

    def __repr__(self):
        u=_SO3weights.view(self.parts)
        return u.__repr__()

    def __str__(self):
        u = _SO3weights.view(self.parts)
        return u.__str__()
