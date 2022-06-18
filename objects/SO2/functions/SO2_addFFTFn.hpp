// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO2_addFFTFn
#define _SO2_addFFTFn

#include "GElib_base.hpp"
#include "CtensorB.hpp"
#include "MultiLoop.hpp"

extern GElib::SO2FourierMatrixBank SO2FmatrixBank;


namespace GElib{

  class SO2_addFFTFn{
  public:

    void operator()(const cnine::Ctensor3_view& _r, const cnine::Ctensor3_view& _x){

    }

  };

}

#endif 

