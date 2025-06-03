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


#ifndef _triple_map
#define _triple_map

#include "Cnine_base.hpp"
#include "Gdims.hpp"


namespace cnine{

  template<typename IX1, typename IX2, typename IX3>
  class triple_index{
  public:

    IX1 i1;
    IX2 i2;
    IX3 i3;

    triple_index(const IX1 _i1, const IX2 _i2, const IX3 _i3): 
      i1(_i1), i2(_i2), i3(_i3){}

    bool operator==(const triple_index& x) const{
      return (i1==x.i1)&&(i2==x.i2)&&(i3==x.i3);
    }

  };
}


namespace std{

  template<typename IX1, typename IX2, typename IX3>
  struct hash<cnine::triple_index<IX1,IX2,IX3> >{
  public:
    size_t operator()(const cnine::triple_index<IX1,IX2,IX3>& x) const{
      size_t h=hash<IX1>()(x.i1);
      h=(h<<1)^hash<IX2>()(x.i2);
      h=(h<<1)^hash<IX3>()(x.i3);
      return h;
    }
  };

}


namespace cnine{
    
  template<typename IX1, typename IX2, typename IX3, typename OBJ>
  class triple_map: public unordered_map<triple_index<IX1,IX2,IX3>,OBJ>{
  public:

    typedef unordered_map<triple_index<IX1,IX2,IX3>,OBJ> BASE;
    typedef triple_index<IX1,IX2,IX3> INDEX;

    OBJ& operator()(const IX1& i1, const IX2& i2, const IX3& i3){
      return BASE::operator[](INDEX(i1,i2,i3));
    }

  };

}

#endif 
