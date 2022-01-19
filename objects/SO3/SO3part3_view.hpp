
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3part3_view
#define _SO3part3_view

//#include "CtensorB.hpp"
#include "Ctensor3_view.hpp"
#include "SO3part2_view.hpp"
//#include "SO3_CGbank.hpp"
//#include "SO3_SPHgen.hpp"
//#include "SO3element.hpp"
//#include "WignerMatrix.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;



namespace GElib{

  class SO3part3_view: public cnine::Ctensor3_view{
  public:

    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::Gdims Gdims;
    typedef cnine::Gstrides Gstrides;

    //float* arr;
    //float* arrc;
    //int n0,n1,n2;
    //int s0,s1,s2;
    int l;
    float* ar;
    float* ac;

  public:


    SO3part3_view(){}

    //SO3part3_view(float* _arr, float* _arrc): 
    //arr(_arr), arrc(_arrc){}

    /*
    SO3part3_view(float* _arr, float* _arrc, const int _n0, const int _n1, const int _n2, 
      const int _s0, const int _s1, const int _s2): 
      arr(_arr), arrc(_arrc), n0(_n0), n1(_n1), n2(_n2), s0(_s0), s1(_s1), s2(_s2){
      assert(n0%2==1);
      l=(n1-1)/2;
      ar=arr+l*s1;
      ac=arrc+l*s1;
    }
    */

    SO3part3_view(float* _arr, const int _n0, const int _n1, const int _n2, 
      const int _s0, const int _s1, const int _s2, const int _coffs=1): 
      Ctensor3_view(_arr,_n0,_n1,_n2,_s0,_s1,_s2,_coffs){
      assert(n1%2==1);
      l=(n1-1)/2;
      ar=arr+l*s1;
      ac=arrc+l*s1;
    }

    SO3part3_view(float* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _coffs=1):
      Ctensor3_view(_arr,_dims,_strides,_coffs){
      assert(n1%2==1);
      l=(n1-1)/2;
      ar=arr+l*s1;
      ac=arrc+l*s1;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    int getl() const{
      return l;
    }

    complex<float> operator()(const int i0, const int m, const int i2){
      int t=s0*i0+s1*m+s2*i2;
      return complex<float>(ar[t],ac[t]);
    }

    void set(const int i0, const int m, const int i2, complex<float> x){
      int t=s0*i0+s1*m+s2*i2;
      ar[t]=std::real(x);
      ac[t]=std::imag(x);
    }

    void inc(const int i0, const int m, const int i2, complex<float> x){
      int t=s0*i0+s1*m+s2*i2;
      ar[t]+=std::real(x);
      ac[t]+=std::imag(x);
    }


  public: // ---- Other views -------------------------------------------------------------------------------

    
    SO3part2_view slice0(const int i) const{
      return SO3part2_view(arr+i*s0,arrc+i*s0,n1,n2,s1,s2);
    }
  

  };





}

#endif 
