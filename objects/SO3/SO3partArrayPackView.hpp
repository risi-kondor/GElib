
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3partArrayPackView
#define _GElibSO3partArrayPackView

#include "GElib_base.hpp"
#include "TensorPackView.hpp"


namespace GElib{

  template<typename TYPE>
  class SO3partArrayPackView: virtual public cnine::TensorPackView<TYPE>{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::CoutLock CoutLock;

    typedef cnine::Gdims Gdims;



    // ---- Constructors -------------------------------------------------------------------------------------


  };

}

#endif
