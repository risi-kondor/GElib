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


  /*
  template<typename GPART, typename = typename std::enable_if<std::is_base_of<Gpart_type, GPART>::value, GPART>::type>
  GPART CGproduct(const GPART& x, const GPART& y, const typename GPART::IrrepIx& l){
    int m=GPART::Group::CGmultiplicity(x.getl(),y.getl(),l);
    GELIB_ASSRT(m>0);
    GPART R=x.zeros_like(l,x.getn()*y.getn());
    R.add_CGproduct(x,y);
    return R;
  }


  template<typename GTYPE, typename = typename 
	   std::enable_if<std::is_base_of<Gtype_type, GTYPE>::value, GTYPE>::type>
  inline GTYPE CGproduct(const GTYPE& x, const GTYPE& y){
    GTYPE R;
    for(auto& p:x.map)
      for(auto& q:y.map)
	GTYPE::Group::for_each_CGcomponent(p.first,q.first,[&](const typename GTYPE::IrrepIx& z, const int m){
	    R.map[z]+=m*p.second*q.second;});
    return R;
  }

  template<typename GTYPE, typename = typename 
	   std::enable_if<std::is_base_of<Gtype_type, GTYPE>::value, GTYPE>::type>
  inline GTYPE CGproduct(const GTYPE& x, const GTYPE& y, const typename GTYPE::IrrepIx& limit){
    GTYPE R;
    for(auto& p:x.map)
      for(auto& q:y.map)
	GTYPE::Group::for_each_CGcomponent(p.first,q.first,[&](const typename GTYPE::IrrepIx& z, const int m){
	    if(z<=limit) R.map[z]+=m*p.second*q.second;});
    return R;
  }
  */
//   template<typename OBJ>
//   OBJ CGproduct(const OBJ& x, const OBJ& y){
//     return x.CGproduct(y); 
//   }

//   template<typename OBJ, typename ARG0>
//   OBJ CGproduct(const OBJ& x, const OBJ& y, const ARG0 arg0){
//     return x.CGproduct(y,arg0); 
//   }

