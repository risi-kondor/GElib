// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _Gtype
#define _Gtype

#include "Tensor.hpp"

namespace GElib{

  template<class GROUP>
  class Gtype: public map<typename GROUP::IrrepIx,int>{
  public:

    typedef typename GROUP::IrrepIx _IrrepIx;

    Gtype(){}

    Gtype(const _IrrepIx& ix, const int m=1){
      (*this)[ix]=m;
    }

  };

  template<typename GROUP>
  inline Gtype<GROUP> tprod(const Gtype<GROUP>& x, const Gtype<GROUP>& y){
    Gtype<GROUP> R;
    for(auto& p:x)
      for(auto& q:y)
	GROUP::for_each_CGcomponent(p.first,q.first,[&](const typename GROUP::IrrepIx& ix, int m){
	    R[ix]+=m*p.second*q.second;
	  });
    return R;
  }

}

#endif 
