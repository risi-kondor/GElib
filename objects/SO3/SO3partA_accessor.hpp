
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineCtensorA_accessor
#define _CnineCtensorA_accessor

#include "CtensorA_accessor"

namespace GElib{


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
