// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2024, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3functions
#define _SO3functions

#include "Gfunctions.hpp"
#include "SO3part.hpp"
#include "SO3type.hpp"
#include "SO3vec.hpp"


namespace GElib{

  //template<typename TYPE>
  //inline spharm(const cnine::TensorView<TYPE>& x, const int l)

}

#endif 



//    inline SO3type CGproduct(const SO3type& x, const SO3type& y, int limit=-1){
//      if(limit==-1) limit=x.highest()+y.highest();
//      SO3type R;
//      for(auto& p:x.map)
// 	for(auto& q:y.map)
// 	  for(int l=abs(p.first-q.first); l<=p.first+q.first && l<=limit; l++)
// 	    R.map[l]+=p.second*q.second;
//       return R;
//    }

//    template<typename TYPE>
//    inline SO3part<TYPE> CGproduct(const SO3part<TYPE>& x, const SO3part<TYPE>& y, const int l){
//      assert(l>=abs(x.getl()-y.getl()) && l<=x.getl()+y.getl());
//      SO3part<TYPE> R=x.zeros_like(l,x.getn()*y.getn());
//      R.add_CGproduct(x,y);
//      return R;
//    }

  //template<typename TYPE>
  //inline SO3vec<TYPE> CGproduct(const SO3vec<TYPE>& x, const SO3vec<TYPE>& y, const int limit){
  // SO3vec<TYPE> R=x.zeros_like(CGproduct)
  //}
