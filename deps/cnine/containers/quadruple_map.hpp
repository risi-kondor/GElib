/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _quadruple_map
#define _quadruple_map

#include "Cnine_base.hpp"
#include "Gdims.hpp"


namespace cnine{

  template<typename IX1, typename IX2, typename IX3, typename IX4>
  class quadruple_index{
  public:

    IX1 i1;
    IX2 i2;
    IX3 i3;
    IX4 i4;

    quadruple_index(const IX1 _i1, const IX2 _i2, const IX3 _i3, const IX4 _i4): 
      i1(_i1), i2(_i2), i3(_i3), i4(_i4){}

    bool operator==(const quadruple_index& x) const{
      return (i1==x.i1)&&(i2==x.i2)&&(i3==x.i3)&&(i4==x.i4);
    }

  };
}


namespace std{

  template<typename IX1, typename IX2, typename IX3, typename IX4>
  struct hash<cnine::quadruple_index<IX1,IX2,IX3,IX4> >{
  public:
    size_t operator()(const cnine::quadruple_index<IX1,IX2,IX3,IX4>& x) const{
      size_t h=hash<IX1>()(x.i1);
      h=(h<<1)^hash<IX2>()(x.i2);
      h=(h<<1)^hash<IX3>()(x.i3);
      h=(h<<1)^hash<IX4>()(x.i4);
      return h;
    }
  };

}


namespace cnine{
    
  template<typename IX1, typename IX2, typename IX3, typename IX4, typename OBJ>
  class quadruple_map: public unordered_map<quadruple_index<IX1,IX2,IX3,IX4>,OBJ>{
  public:

    typedef unordered_map<quadruple_index<IX1,IX2,IX3,IX4>,OBJ> BASE;
    typedef quadruple_index<IX1,IX2,IX3,IX4> INDEX;

    OBJ& operator()(const IX1& i1, const IX2& i2, const IX3& i3, const IX4& i4){
      return BASE::operator[](INDEX(i1,i2,i3,i4));
    }

  };

}

#endif 
