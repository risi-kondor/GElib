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
#from gelib_base import add_WignerMatrix
#from gelib_base import SO3Fvec as _SO3Fvec

import gelib_base

from gelib import *


def WignerMatrix(l,phi,theta,psi,_dev=0):
    if _dev==0:
        r=torch.zeros(2*l+1,2*l+1,2)
    else:
        r=torch.zeros(2*l+1,2*l+1,2).cuda()
    _r=ctensorb.view(r)
    gelib_base.add_WignerMatrix_to(_r,l,phi,theta,psi)
    return r

