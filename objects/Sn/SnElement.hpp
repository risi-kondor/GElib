
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SnElement
#define _SnElement

#include "Permutation.hpp"

namespace GElib{


  class SnElement: public Permutation{
  public:

    using Permutation::Permutation;


  public: // conversions 

    SnElement(const Permutation& x):
      Permutation(x){}


  public: // named constructors 

    static SnElement Identity(const int n){
      return Permutation(n,cnine::fill_identity());
    }


  };

}

#endif
