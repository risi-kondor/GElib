# This file is part of GElib, a C++/CUDA library for group
# equivariant tensor operations. 
# 
# Copyright (c) 2022, Imre Risi Kondor
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch

from cnine import rtensor 
from gelib_base import add_CGtensor_to, add_CGmatrix_to

def SO3CGtensor(l1,l2,l):
    r=torch.zeros(2*l1+1,2*l2+1,2*l+1)
    add_CGtensor_to(rtensor.view(r),l1,l2,l)
    return r

def SO3CGmatrix(l1,l2,l):
    r=torch.zeros(2*l1+1,2*l2+1)
    add_CGmatrix_to(rtensor.view(r),l1,l2,l)
    return r


