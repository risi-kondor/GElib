
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3_CGindex
#define _SO3_CGindex

#include "GElib_base.hpp"


namespace GElib{

  class SO3CGindex{
  public:

    int l1,l2,l;

    SO3CGindex(const int _l1, const int _l2, const int _l): 
      l1(_l1), l2(_l2), l(_l){
      assert(l1>=0); assert(l2>=0); assert(l>=0);
      assert(l<=l1+l2); assert(l>=abs(l1-l2));
    }

    bool operator==(const SO3CGindex& x) const{
      return (l1==x.l1)&&(l2==x.l2)&&(l==x.l);}

    string str() const{
      return "("+to_string(l1)+","+to_string(l2)+","+to_string(l)+")";}

  };

} 


namespace std{
  template<>
  struct hash<GElib::SO3CGindex>{
  public:
    size_t operator()(const GElib::SO3CGindex& ix) const{
      return ((hash<int>()(ix.l1)<<1)^hash<int>()(ix.l2)<<1)^hash<int>()(ix.l);}
  };
}

#endif 
