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


#ifndef _double_map
#define _double_map

#include "Cnine_base.hpp"
//#include "Gdims.hpp"


namespace cnine{

  template<typename IX1, typename IX2>
  class double_index{
  public:

    IX1 i1;
    IX2 i2;

    double_index(const IX1 _i1, const IX2 _i2): 
      i1(_i1), i2(_i2){}

    bool operator<(const double_index y) const{
      if(i1<y.i1) return true;
      if(i1>y.i1) return false;
      if(i2<y.i2) return true;
      return false;
    }

    bool operator==(const double_index& x) const{
      return (i1==x.i1)&&(i2==x.i2);
    }
    
  };

}


namespace std{

  template<typename IX1, typename IX2>
  struct hash<cnine::double_index<IX1,IX2> >{
  public:
    size_t operator()(const cnine::double_index<IX1,IX2>& x) const{
      size_t h=hash<IX1>()(x.i1);
      h=(h<<1)^hash<IX2>()(x.i2);
      return h;
    }
  };

}


namespace cnine{
    
  template<typename IX1, typename IX2, typename OBJ>
  class double_map: public unordered_map<double_index<IX1,IX2>,OBJ>{
  public:

    typedef unordered_map<double_index<IX1,IX2>,OBJ> BASE;
    typedef double_index<IX1,IX2> INDEX;

    OBJ& operator()(const IX1& i1, const IX2& i2){
      return BASE::operator[](INDEX(i1,i2));
    }

  };

}

#endif 
