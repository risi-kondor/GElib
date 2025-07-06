# This file is part of GElib, a C++/CUDA library for group
# equivariant tensor operations. 
# 
# Copyright (c) 2024, Imre Risi Kondor
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


import torch
import gelib_base as gb
from gelib import *


class O3irrep:
    """
    An element of the group SO(3)
    """

    def __init__(self, l,p):
        assert(isinstance(l,int))
        assert(isinstance(p,int))
        self.obj=gb.O3irrep(l,p)
        

    # ---- Static constructors -----------------------------------------------------------------------------



    # ---- Operations --------------------------------------------------------------------------------------


    def matrix(self,R):
        if isinstance(R,O3element):
            return self.obj.matrix(gb.O3element.view(R))
        if isinstance(R,list):
            assert len(R)==3
            return self.obj.matrix(R[0],R[1],R[2])



    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    
