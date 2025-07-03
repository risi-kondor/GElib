# This file is part of GElib, a C++/CUDA library for group equivariant 
# tensor operations. 
#  
# Copyright (c) 2023, Imre Risi Kondor
#
# This source code file is subject to the terms of the noncommercial 
# license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
# use is prohibited. All redistributed versions of this file (in orginal
# or modified form) must retain this copyright notice and must be 
# accompanied by a verbatim copy of the license. 

import torch
import gelib_base as gb


class SO3element(torch.Tensor):
    """
    An element of the group SO(3)
    """

    def __init__(self, _T):
        self=_T
        #super().__init__()


    # ---- Static constructors -----------------------------------------------------------------------------


    @classmethod
    def identity(self):
        return SO3element(torch.eye(3))

    @classmethod
    def random(self):
        return SO3element(gb.SO3element.random().torch())


    # ---- Operations --------------------------------------------------------------------------------------


    def __mul__(self,y):
        assert(isinstance(y,SO3element))
        return SO3element(torch.matmul(self,y))

    def inv(self):
        return SO3element(self.t())


    # ---- I/O ----------------------------------------------------------------------------------------------
