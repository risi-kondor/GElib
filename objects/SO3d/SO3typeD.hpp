// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _SO3typeD
#define _SO3typeD

#include "SO3group.hpp"
#include "GtypeD.hpp"

namespace GElib{

  class SO3typeD: public GtypeD<SO3group>{
  public:

    typedef GtypeD<SO3group> BASE;

    using BASE::BASE;

    SO3typeD(const BASE& x):
      BASE(x){}


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::SO3typeD";
    }


  };

}

#endif 
