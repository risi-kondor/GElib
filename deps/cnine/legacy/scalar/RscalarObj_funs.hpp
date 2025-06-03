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


#ifndef _RscalarObj_funs
#define _RscalarObj_funs

#include "RscalarObj.hpp"


namespace cnine{


  RscalarObj abs(const RscalarObj& x){
    RscalarObj R(fill::zero);
    R.add_abs(x);
    return R;
  }


  RscalarObj norm2(const RscalarObj& x){
    RscalarObj R(fill::zero);
    R.add_prod(x,x);
    return R;
  }


  RscalarObj ReLU(const RscalarObj& x, const float c=0){
    RscalarObj R(fill::zero);
    R.add_ReLU(x,c);
    return R;
  }


  RscalarObj sigmoid(const RscalarObj& x){
    RscalarObj R(fill::zero);
    R.add_sigmoid(x);
    return R;
  }


  RscalarObj inp(const RscalarObj& x, const RscalarObj& y){
    RscalarObj R(fill::zero);
    R.add_prod(x,y);
    return R;
  }


}

#endif 


  // ---- In-place operators ---------------------------------------------------------------------------------

  /*
  RscalarObj& operator+=(RscalarObj& x, const RscalarObj& y){
    x.add(y);
    return x;
  }

  RscalarObj& operator-=(RscalarObj& x, const RscalarObj& y){
    x.subtract(y);
    return x;
  }
  */


  /*
  RscalarObj sum(const ListOf<RscalarObj>& v){
    const int N=v.obj.size();
    if(N==0) return 0;
    RscalarObj r(fill::zero,v.obj[0]->nbu);
    r.add_sum(v.obj);
    return r;
  }
  */
