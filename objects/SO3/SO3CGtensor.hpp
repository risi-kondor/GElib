// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _SO3CGtensor
#define _SO3CGtensor

#include "GElib_base.hpp"
#include "RtensorA.hpp"

extern GElib::SO3_CGbank SO3_cgbank;


namespace GElib{

  cnine::RtensorA SO3CGmatrix(const int l1, const int l2, const int l){
    return cnine::RtensorA(SO3_cgbank.getf(CGindex(l1,l2,l)));
  }

  cnine::RtensorA SO3CGtensor(const int l1, const int l2, const int l){
    cnine::RtensorA R=cnine::RtensorA::zero({2*l1+1,2*l2+1,2*l+1});
    auto C=SO3_cgbank.getf(CGindex(l1,l2,l));
    int a=l1+l2-l;
    for(int m1=0; m1<2*l1+1; m1++)
      for(int m2=0; m2<2*l2+1; m2++)
	R.set(m1,m2,m1+m2-a,C(m1,m2));
    return R;
  }


}

#endif 
