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
from . import gelib_base as gb

class gather_map:

    def __init__(self, x):
        assert isinstance(x,gb.gather_map)
        self.obj=x

    @classmethod
    def from_matrix(self,M,n_in,n_out):
        assert isinstance(M,torch.Tensor)
        assert isinstance(n_in,int)
        assert isinstance(n_out,int)
        return gather_map(gb.gather_map.from_matrix(x.int(),n_in,n_out))

    @classmethod
    def random(self,n_in,n_out,p):
        return gather_map(gb.gather_map.random(n_in,n_out,p))



    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()

