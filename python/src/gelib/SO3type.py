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


class SO3type:

    def __init__(self,x):
        if isinstance(x,dict):
            self.obj=gb.SO3type(x)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()



    
