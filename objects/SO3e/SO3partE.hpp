// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _SO3partE
#define _SO3partE

#include "SO3groupE.hpp"
#include "GpartSpec.hpp"
#include "GpartE.hpp"


namespace GElib{


  template<typename TYPE>
  class SO3partE: public GpartE<complex<TYPE> >{
  public:

    typedef GpartE<complex<TYPE> > BASE;
    using BASE::BASE;

    SO3partE(const BASE& x):
      BASE(x){}


  public: // ---- SO3partSpec -------------------------------------------------------------------------------


    SO3partE(const SO3partSpec<TYPE>& x):
      BASE(x){}

    static SO3partSpec<TYPE> raw() {return SO3partSpec<TYPE>().raw();}
    static SO3partSpec<TYPE> zero() {return SO3partSpec<TYPE>().zero();}
    static SO3partSpec<TYPE> sequential() {return SO3partSpec<TYPE>().sequential();}
    static SO3partSpec<TYPE> gaussian() {return SO3partSpec<TYPE>().gaussian();}


  };

}

#endif 

