// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _Gisotypic
#define _Gisotypic

#include "cachedf.hpp"


namespace GElib{

  template<typename Group, typename TYPE>
  class Gisotypic: public Tensor<TYPE>{
  public:

    typedef typename GROUP::IrrepIx _IrrepIx;

    _IrrepIx ix;


    

  };


}

#endif 
