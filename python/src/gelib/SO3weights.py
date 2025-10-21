# This file is part of GElib, a C++/CUDA library for group equivariant 
# tensor operations. 
#  
# Copyright (c) 2025, Imre Risi Kondor
#
# This source code file is subject to the terms of the noncommercial 
# license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
# use is prohibited. All redistributed versions of this file (in orginal
# or modified form) must retain this copyright notice and must be 
# accompanied by a verbatim copy of the license. 

import torch
import gelib_base as gb
from gelib import *


# ----------------------------------------------------------------------------------------------------------
# ---- SO3weights ------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3weights:
    """
    A sequence of weight matrices to be used for multyplying an SO3vec or SO3vecArr with in an equivariant way
    """

    def __init__(self,*args):
        if not args:
            self.parts=[]
            return 
        self.parts=args

    # ---- Static constructors ------------------------------------------------------------------------------


    @classmethod
    def zeros(self,tau_in,tau_out,device='cpu'):
        "Construct a zero SO3weight."
        assert(isinstance(tau_in,SO3type))
        assert(isinstance(tau_out,SO3type))
        R=SO3weights()
        for l,n in tau_in.items():
            R.parts.append(torch.zeros([tau_in[l],tau_out[l]],dtype=torch.complex64,device=device))
        return R

    @classmethod
    def randn(self,tau_in,tau_out,device='cpu'):
        "Construct a random SO3weight."
        assert(isinstance(tau_in,SO3type))
        assert(isinstance(tau_out,SO3type))
        R=SO3weights()
        for l,n in tau_in.items():
            R.parts.append(torch.randn([tau_in[l],tau_out[l]],dtype=torch.complex64,device=device))
        return R


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        r="<SO3weights "
        for x in self.parts:
            r+="("+str(x.size(0))+","+str(x.size(1))+"),"
        r+=">"
        return r

    def __str__(self):
        r=""
        for x in self.parts:
            r+=str(x)+"\n"
        return r
    
