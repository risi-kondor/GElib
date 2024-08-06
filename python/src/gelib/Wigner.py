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
from gelib_base import add_WignerMatrix_to


def WignerMatrix(l,phi,theta,psi,_dev=0):
    if _dev==0:
        r=torch.zeros(2*l+1,2*l+1,dtype=torch.cfloat)
    else:
        r=torch.zeros(2*l+1,2*l+1,dtype=torch.cfloat).cuda()
    _r=ctensorb.view(r)
    add_WignerMatrix_to(_r,l,phi,theta,psi)
    return r

