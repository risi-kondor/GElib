
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3partArrayView
#define _GElibSO3partArrayView

#include "GElib_base.hpp"
#include "TensorView.hpp"
#include "SO3partView.hpp"

namespace GElib{

  template<typename RTYPE>
  class SO3partArrayView: public cnine::TensorView<complex<RTYPE> >{
  public:
    
  public: // ---- Access --------------------------------------------------------------------------------------
    
  };

}


#endif 
