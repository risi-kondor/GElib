// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3irrepAction
#define _SO3irrepAction

#include "GElib_base.hpp"
#include "CtensorB.hpp"
#include "SO3element.hpp"
#include "WignerMatrix.hpp"


namespace GElib{

  class SO3irrepAction{
  public:

    int l;
    
    SO3irrepAction(const int _l):
      l(_l){}

    CtensorB operator(const CtensorB& x, const SO3element& r, const int d=0){
      inte dev=x.dev;
      CtensorB D(WignerMatrix<float>(getl(),r),dev);
      CtensorB R=CtensorB::zeros_like(x);
      R.add_matmul(x,D,d);
      return R;
    }

  };


}

#endif
