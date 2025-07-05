// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2024, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _Gfunctions
#define _Gfunctions

#include "Gpart.hpp"


namespace GElib{


  template<typename OBJ, typename... Args>
  OBJ CGproduct(const OBJ& x, const OBJ& y, const Args&... args){
    return x.CGproduct(y,args...); 
  }

  template<typename OBJ, typename... Args>
  OBJ DiagCGproduct(const OBJ& x, const OBJ& y, const Args&... args){
    return x.DiagCGproduct(y,args...); 
  }

}

#endif 


