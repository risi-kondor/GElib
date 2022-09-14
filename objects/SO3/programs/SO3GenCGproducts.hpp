
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3GenCGproducts
#define _SO3GenCGproducts

#include "SO3vecB.hpp"
#include "SO3CGprogramBank.hpp"
#include "SO3CGexec.hpp"

extern GElib::SO3CGprogramBank SO3_GGprogram_bank;


namespace GElib{


  void add_CGproduct(SO3vecB& R, const std::vector<const SO3vecB*>& v , const int maxl=-1){
    const SO3CGprogram& prg=SO3_CGprogram_bank.add_CGproduct(get_types(v),maxl);
    SO3CGexec frame(prg);
    frame(R,v);
  }


  inline SO3vecB CGproduct(const std::vector<const SO3vecB*>& v , const int maxl=-1){
    assert(v.size()>0);
    const int B=v[0]->getb();
    SO3vecB R=SO3vecB::zero(B,CGproduct(get_types(v),maxl));
    add_CGproduct(R,v,maxl);
    return R;
  }


  inline SO3vecB CGproduct(const initializer_list<const SO3vecB>& v , const int maxl=-1){
    std::vector<const SO3vecB*> w;
    for(auto& p: v)
      w.push_back(&p);
    return CGproduct(w, maxl);
  }


}

#endif 
