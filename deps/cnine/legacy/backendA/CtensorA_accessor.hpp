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


#ifndef _CnineCtensorA_accessor
#define _CnineCtensorA_accessor

#include "PretendComplex.hpp"

namespace cnine{


  class CtensorA_accessor{
  public:

    float* arr;
    float* arrc;
    const vector<int>& strides;

  public:

    CtensorA_accessor(float* _arr, float* _arrc, const vector<int>& _strides):
      arr(_arr), arrc(_arrc), strides(_strides){
    }

  public:

    pretend_complex<float> operator[](const int t){
      return pretend_complex<float>(arr+t,arrc+t);
    }

    complex<float> geti(const int t){
      return complex<float>(arr[t],arrc[t]);
    }

    void seti(const int t, complex<float> x){
      arr[t]=std::real(x);
      arrc[t]=std::imag(x);
    }

    void inci(const int t, complex<float> x){
      arr[t]+=std::real(x);
      arrc[t]+=std::imag(x);
    }

  public:

    complex<float> operator()(const int i0, const int i1){
      int t=strides[0]*i0+strides[1]*i1;
      return complex<float>(arr[t],arrc[t]);
    }

    void set(const int i0, const int i1, complex<float> x){
      int t=strides[0]*i0+strides[1]*i1;
      arr[t]=std::real(x);
      arrc[t]=std::imag(x);
    }

    void inc(const int i0, const int i1, complex<float> x){
      int t=strides[0]*i0+strides[1]*i1;
      arr[t]+=std::real(x);
      arrc[t]+=std::imag(x);
    }

  public:


  };

}

#endif 
