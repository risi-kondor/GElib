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


class SO3irrep:
    """
    An element of the group SO(3)
    """

    def __init__(self, l):
        assert(isinstance(l,int))
        self.obj=gb.SO3irrep(l)
        

    # ---- Static constructors -----------------------------------------------------------------------------



    # ---- Operations --------------------------------------------------------------------------------------


    def matrix(self,R):
        assert(isinstance(R,SO3element))
        return self.obj.matrix(gb.SO3element.view(R))


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    
