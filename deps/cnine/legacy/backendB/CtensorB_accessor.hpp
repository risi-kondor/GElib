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


#ifndef _CnineCtensorB_accessor
#define _CnineCtensorB_accessor

#include "Gstrides.hpp"

namespace cnine{


  class CtensorB_accessor1{
  public:

    float* arr;
    int s0;

  public:

    CtensorB_accessor1(const Gstrides& strides){
      assert(strides.size()==1);
      s0=strides[0];
    }


  public: 

    complex<float> operator()(const int i0){
      int t=s0*i0;
      return complex<float>(arr[t],arr[t+1]);
    }

    void set(const int i0, complex<float> x){
      int t=s0*i0;
      arr[t]=std::real(x);
      arr[t+1]=std::imag(x);
    }

    void inc(const int i0, complex<float> x){
      int t=s0*i0;
      arr[t]+=std::real(x);
      arr[t+1]+=std::imag(x);
    }

  };


  class CtensorB_accessor2{
  public:

    float* arr;
    int s0,s1;

  public:

    CtensorB_accessor2(const Gstrides& strides){
      assert(strides.size()==2);
      s0=strides[0];
      s1=strides[1];
    }


  public: 

    complex<float> operator()(const int i0, const int i1){
      int t=s0*i0+s1*i1;
      return complex<float>(arr[t],arr[t+1]);
    }

    void set(const int i0, const int i1, complex<float> x){
      int t=s0*i0+s1*i1;
      arr[t]=std::real(x);
      arr[t+1]=std::imag(x);
    }

    void inc(const int i0, const int i1, complex<float> x){
      int t=s0*i0+s1*i1;
      arr[t]+=std::real(x);
      arr[t+1]+=std::imag(x);
    }

  public:

    


  };


  class CtensorB_accessor3{
  public:

    float* arr;
    int s0,s1,s2;

  public:

    CtensorB_accessor3(const Gstrides& strides){
      assert(strides.size()==3);
      s0=strides[0];
      s1=strides[1];
      s2=strides[2];
    }


  public: 

    complex<float> operator()(const int i0, const int i1, const int i2){
      int t=s0*i0+s1*i1+s2*i2;
      return complex<float>(arr[t],arr[t+1]);
    }

    void set(const int i0, const int i1, const int i2, complex<float> x){
      int t=s0*i0+s1*i1+s2*i2;
      arr[t]=std::real(x);
      arr[t+1]=std::imag(x);
    }

    void inc(const int i0, const int i1, const int i2, complex<float> x){
      int t=s0*i0+s1*i1+s2*i2;
      arr[t]+=std::real(x);
      arr[t+1]+=std::imag(x);
    }

  };

}


#endif 
