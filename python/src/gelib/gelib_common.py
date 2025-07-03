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
from gelib import *


def common_batch(x,y):
    assert x.dim()>=1
    assert y.dim()>=1
    if x.size(0)>1 and y.size(0)>1:
        assert(x.size(0)==y.size(0))
    return max(x.size(0),y.size(0))

