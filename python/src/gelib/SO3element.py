# This file is part of GElib, a C++/CUDA library for group
# equivariant tensor operations. 
# 
# Copyright (c) 2022, Imre Risi Kondor
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

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
