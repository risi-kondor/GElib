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


#ifndef _CnineRtensorA_accessor
#define _CnineRtensorA_accessor

#include "PretendComplex.hpp"

namespace cnine{


  class RtensorA_accessor{
  public:

    float* arr;
    const vector<int>& strides;

  public:

    RtensorA_accessor(float* _arr, const vector<int>& _strides):
      arr(_arr), strides(_strides){
    }

  public:

    float& operator[](const int t){
      return arr[t];
    }

    float geti(const int t){
      return arr[t];
    }

    void seti(const int t, float x){
      arr[t]=x;
    }

    void inci(const int t, float x){
      arr[t]+=x;
    }

  public:

    float operator()(const int i0, const int i1){
      int t=strides[0]*i0+strides[1]*i1;
      return arr[t];
    }

    void set(const int i0, const int i1, float x){
      int t=strides[0]*i0+strides[1]*i1;
      arr[t]=x;
    }

    void inc(const int i0, const int i1, float x){
      int t=strides[0]*i0+strides[1]*i1;
      arr[t]+=x;
    }

  public:


  };

}

#endif 
