
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3Fpart2_view
#define _SO3Fpart2_view

//#include "CtensorB.hpp"
#include "Ctensor2_view.hpp"
//#include "SO3_CGbank.hpp"
//#include "SO3_SPHgen.hpp"
//#include "SO3element.hpp"
//#include "WignerMatrix.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;



namespace GElib{

  class SO3Fpart2_view: public cnine::Ctensor2_view{
  public:

    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::Gdims Gdims;
    typedef cnine::Gstrides Gstrides;

    int l;
    float* ar;
    float* ac;

  public:


    SO3Fpart2_view(){}

    //SO3Fpart3_view(float* _arr, float* _arrc): 
    //arr(_arr), arrc(_arrc){}

    
    SO3Fpart2_view(float* _arr, float* _arrc, const int _n0, const int _n1,  const int _s0, const int _s1): 
      Ctensor2_view(_arr,_arrc,_n0,_n1,_s0,_s1){
      //arr(_arr), arrc(_arrc), n0(_n0), n1(_n1), s0(_s0), s1(_s1){
      assert(n1==n0);
      assert(n0%2==1);
      l=(n0-1)/2;
      ar=arr+l*s0+l*s1;
      ac=arrc+l*s0+l*s1;
    }

    SO3Fpart2_view(float* _arr, const int _n0, const int _n1,  
      const int _s0, const int _s1, const int _coffs=1): 
      Ctensor2_view(_arr,_n0,_n1,_s0,_s1,_coffs){
      assert(n1==n0);
      assert(n0%2==1);
      l=(n0-1)/2;
      ar=arr+l*s0+l*s1;
      ac=arrc+l*s0+l*s1;
    }

    SO3Fpart2_view(float* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _coffs=1):
      Ctensor2_view(_arr,_dims,_strides,_coffs){
      assert(n1==n0);
      assert(n0%2==1);
      l=(n0-1)/2;
      ar=arr+l*s0+l*s1;
      ac=arrc+l*s0+l*s1;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    int getl() const{
      return l;
    }

    complex<float> operator()(const int m0, const int m1){
      int t=s0*m0+s1*m1;
      return complex<float>(ar[t],ac[t]);
    }

    void set(const int m0, const int m1, complex<float> x){
      int t=s0*m0+s1*m1;
      ar[t]=std::real(x);
      ac[t]=std::imag(x);
    }

    void inc(const int m0, const int m1, complex<float> x){
      int t=s0*m0+s1*m1;
      ar[t]+=std::real(x);
      ac[t]+=std::imag(x);
    }


  public: // ---- Other views -------------------------------------------------------------------------------

    
    //SO3part2_view slice0(const int i) const{
    //return SO3part2_view(arr+i*s0,arrc+i*s0,n1,n2,s1,s2);
    //}

  
  public: // ---- Operations ---------------------------------------------------------------------------------


    SO3Fpart2_view& flip(){
      arr=arr+(n0-1)*s0+(n1-1)*s1;
      arrc=arrc+(n0-1)*s0+(n1-1)*s1;
      ar=arr-l*s0-l*s1;
      ac=arrc-l*s0-l*s1;
      s0=-s0;
      s1=-s1;
      return *this;
    }

  };





}

#endif 
