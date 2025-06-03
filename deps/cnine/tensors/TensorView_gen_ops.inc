/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021--, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */



// ---- Not in-place operations  -----------------------------------------------------------------------------


TensorView odot(const TensorView& y) const{
  auto r=zeros_like();
  TYPE t=0;
  zip(y,r,[&t](TYPE& a, TYPE& b, TYPE& c){c=a*b;});
  return r;
}

TensorView sum(const int d) const{
  TensorView R(dims.remove(d),0,dev);
  R.add_sum(d,*this);
  return R;
}

TensorView times(const TYPE c) const{
  auto r=zeros_like();
  r.add(*this,c);
  return r;
}

TensorView operator*(const TYPE c) const{
  return times(c);
}



