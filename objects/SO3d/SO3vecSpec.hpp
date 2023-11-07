// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3vecSpec
#define _GElibSO3vecSpec

#include "GElib_base.hpp"
#include "GvecSpec.hpp"
#include "SO3type.hpp"
#include "SO3typeD.hpp"

// can we just use GvecSpec for this?

namespace GElib{

  template<typename TYPE>
  class SO3vecSpec: public GvecSpec<SO3vecSpec<TYPE>,SO3typeD >{
  public:

    typedef GvecSpec<SO3vecSpec<TYPE>,SO3typeD > BASE;

    using BASE::BASE;

    //using BASE::grid;

    SO3vecSpec(const BASE& x): 
      BASE(x){}


  };

}

#endif 
