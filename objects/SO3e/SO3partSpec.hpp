// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _SO3partSpec
#define _SO3partSpec

#include "SO3groupE.hpp"
#include "GpartSpec.hpp"

namespace GElib{

  template<typename TYPE>
  class SO3partE;


  template<typename TYPE>
  class SO3partSpec: public GpartSpecBase<SO3partSpec<TYPE> >{
  public:

    typedef GpartSpecBase<SO3partSpec<TYPE> > BASE;

    using BASE::ddims;
    using BASE::ix;

    SO3partSpec():
      BASE(new SO3group()){}

    SO3partSpec(const BASE& x): 
      BASE(x){
      if(ddims.size()!=2) ddims=cnine::Gdims(0,0);
    }

    SO3partE<TYPE> operator ()() const{
      return SO3partE<TYPE>(*this);
    }

    SO3partSpec& l(const int _l){
      ix.reset(new SO3irrepIx(_l));
      ddims[0]=2*_l+1;
      return *this;
    }

  };

}

#endif 
