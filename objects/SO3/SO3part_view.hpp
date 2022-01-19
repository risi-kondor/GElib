
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3part2_view
#define _SO3part2_view

#include "CtensorB.hpp"
#include "SO3_CGbank.hpp"
#include "SO3_SPHgen.hpp"
#include "SO3element.hpp"
#include "WignerMatrix.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;



namespace GElib{

  class SO3part2_view{
  public:

    float* arr;
    float* arrc;
    int n0,n1;
    int s0,s1;
    int l;
    float* ar;
    float* ac;

  public:


    SO3part2_view(){}

    SO3part2_view(float* _arr, float* _arrc): 
      arr(_arr), arrc(_arrc){}

    SO3part2_view(float* _arr, float* _arrc, const int _n0, const int _n1, const int _s0, const int _s1): 
      arr(_arr), arrc(_arrc), n0(_n0), n1(_n1), s0(_s0), s1(_s1){
      assert(n0%2==1);
      l=(n0-1)/2;
      ar=arr-l;
      ac=arrc-l;
    }

    SO3part2_view(float* _arr, const int _n0, const int _n1, const int _s0, const int _s1, const int _coffs=1): 
      arr(_arr), arrc(_arr+_coffs), n0(_n0), n1(_n1), s0(_s0), s1(_s1){
      assert(n0%2==1);
      l=(n0-1)/2;
      ar=arr-l;
      ac=arrc-l;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    complex<float> operator()(const int m, const int i1){
      int t=s0*m+s1*i1;
      return complex<float>(ar[t],ac[t]);
    }

    void set(const int m, const int i1, complex<float> x){
      int t=s0*m+s1*i1;
      ar[t]=std::real(x);
      ac[t]=std::imag(x);
    }

    void inc(const int m, const int i1, complex<float> x){
      int t=s0*m+s1*i1;
      ar[t]+=std::real(x);
      ac[t]+=std::imag(x);
    }

  };





}

#endif 
