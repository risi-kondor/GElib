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


// ---- Getters ------------------------------------------------------------------------------------


TYPE operator()(const Gindex& ix) const{
  CNINE_CHECK_RANGE(dims.check_in_range(ix,string(__PRETTY_FUNCTION__)));
  return arr[strides.offs(ix)];
}

TYPE operator()(const int i0) const{
  CNINE_CHECK_RANGE(dims.check_in_range(i0,string(__PRETTY_FUNCTION__)));
  return arr[strides.offs(i0)];
}

TYPE& operator()(const int i0, const int i1) const{
  CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,string(__PRETTY_FUNCTION__)));
  return arr[strides.offs(i0,i1)];
}

TYPE operator()(const int i0, const int i1, const int i2) const{
  CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,string(__PRETTY_FUNCTION__)));
  return arr[strides.offs(i0,i1,i2)];
}

TYPE operator()(const int i0, const int i1, const int i2, const int i3) const{
  CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,i3,string(__PRETTY_FUNCTION__)));
  return arr[strides.offs(i0,i1,i2,i3)];
}


TYPE get(const int i0) const{
  return operator()(i0);
}

TYPE get(const int i0, const int i1) const{
  return operator()(i0,i1);
}

TYPE get_value(const int i0) const{
  return operator()(i0);
}

TYPE get_value(const int i0, const int i1) const{
  return operator()(i0,i1);
}

TYPE get_value(const int i0, const int i1, const int i2) const{
  return operator()(i0,i1,i2);
}


TYPE& operator()(const int i0){
  CNINE_CHECK_RANGE(dims.check_in_range(i0,string(__PRETTY_FUNCTION__)));
  return arr[strides.offs(i0)];
}


// ---- Setters ------------------------------------------------------------------------------------


TYPE set(const Gindex& ix, const TYPE x) const{
  CNINE_CHECK_RANGE(dims.check_in_range(ix,string(__PRETTY_FUNCTION__)));
  arr[strides.offs(ix)]=x;
  return x;
}

TYPE set(const int i0, const TYPE x) const{
  CNINE_CHECK_RANGE(dims.check_in_range(i0,string(__PRETTY_FUNCTION__)));
  arr[strides.offs(i0)]=x;
  return x;
}

TYPE set(const int i0, const int i1,  const TYPE x) const{
  CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,string(__PRETTY_FUNCTION__)));
  arr[strides.offs(i0,i1)]=x;
  return x;
}

TYPE set(const int i0, const int i1, const int i2, const TYPE x) const{
  CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,string(__PRETTY_FUNCTION__)));
  arr[strides.offs(i0,i1,i2)]=x;
  return x;
}

TYPE set(const int i0, const int i1, const int i2, const int i3, const TYPE x) const{
  CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,i3,string(__PRETTY_FUNCTION__)));
  arr[strides.offs(i0,i1,i2,i3)]=x;
  return x;
}


// ---- Incrementers -------------------------------------------------------------------------------


void inc(const Gindex& ix, const TYPE x) const{
  CNINE_CHECK_RANGE(dims.check_in_range(ix,string(__PRETTY_FUNCTION__)));
  arr[strides.offs(ix)]+=x;
}

void inc(const int i0, const TYPE x) const{
  CNINE_CHECK_RANGE(dims.check_in_range(i0,string(__PRETTY_FUNCTION__)));
  arr[strides.offs(i0)]+=x;
}

void inc(const int i0, const int i1,  const TYPE x) const{
  CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,string(__PRETTY_FUNCTION__)));
  arr[strides.offs(i0,i1)]+=x;
}

void inc(const int i0, const int i1, const int i2, const TYPE x) const{
  CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,string(__PRETTY_FUNCTION__)));
  arr[strides.offs(i0,i1,i2)]+=x;
}

void inc(const int i0, const int i1, const int i2, const int i3, const TYPE x) const{
  CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,i3,string(__PRETTY_FUNCTION__)));
  arr[strides.offs(i0,i1,i2,i3)]+=x;
}


