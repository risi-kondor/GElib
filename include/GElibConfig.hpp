// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibConfig
#define _GElibConfig

#include "GElib_base.hpp"


namespace GElib{

  class GElibConfig{
  public:

    bool SO3part_CGkernels_explicit=true;

    
  };

}


#endif 
