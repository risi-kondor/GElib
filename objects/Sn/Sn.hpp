
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _Sn
#define _Sn

#include "SnBank.hpp"


namespace GElib{

  class Sn{
  public:

    const int n;
    SnObj* obj;

    Sn(const int _n): n(_n){
      obj=_snbank->get_Sn(n);
    }

  public: // Access

    int size() const{
      return obj->order;
    }

    int get_order() const{
      return obj->order;
    }

    SnElement identity() const{
      return SnElement(n,cnine::fill_identity());
    }

    SnElement element(const int i) const{
      return obj->element(i);
    }

  };

}

#endif 
