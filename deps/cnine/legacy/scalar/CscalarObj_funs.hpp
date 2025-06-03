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


#ifndef _CnineCscalarObj_funs
#define _CnineCscalarObj_funs

#include "CscalarObj.hpp"

namespace cnine{


  RscalarObj real(const CscalarObj& x){
    return x.real();
  }

  RscalarObj imag(const CscalarObj& x){
    return x.imag();
  }

  CscalarObj abs(const CscalarObj& x){
    CscalarObj R(fill::zero);
    R.add_abs(x);
    return R;
  }

  CscalarObj ReLU(const CscalarObj& x, const float c=0){
    CscalarObj R(fill::zero);
    R.add_ReLU(x,c);
    return R;
  }

  CscalarObj inp(const CscalarObj& x, const CscalarObj& y){
    CscalarObj R(fill::zero);
    R.add_prodc(x,y);
    return R;
  }

  CscalarObj operator*(const float c, const CscalarObj& x){
    return x*c;
  }

  CscalarObj operator*(const complex<float> c, const CscalarObj& x){
    return x*c; 
  }



}

#endif 
  //Conjugate<CscalarObj> conj(const CscalarObj& x){
  //return Conjugate<CscalarObj>(x);
  //}
  
  /*
  CscalarObj sum(const ListOf<CscalarObj>& v){
    return CscalarObj::sum(v.obj);
    }
  */
