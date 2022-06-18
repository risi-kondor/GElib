
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3Fpart3_view
#define _SO3Fpart3_view

//#include "CtensorB.hpp"
#include "Ctensor3_view.hpp"
#include "SO3Fpart2_view.hpp"
//#include "SO3_CGbank.hpp"
//#include "SO3_SPHgen.hpp"
//#include "SO3element.hpp"
//#include "WignerMatrix.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;



namespace GElib{

  class SO3Fpart3_view: public cnine::Ctensor3_view{
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


    SO3Fpart3_view(){}

    //SO3Fpart3_view(float* _arr, float* _arrc): 
    //arr(_arr), arrc(_arrc){}

    /*
    SO3Fpart3_view(float* _arr, float* _arrc, const int _n0, const int _n1, const int _n2, 
      const int _s0, const int _s1, const int _s2): 
      arr(_arr), arrc(_arrc), n0(_n0), n1(_n1), n2(_n2), s0(_s0), s1(_s1), s2(_s2){
      assert(n1%2==1);
      l=(n1-1)/2;
      ar=arr+l*s1;
      ac=arrc+l*s1;
    }
    */

    SO3Fpart3_view(float* _arr, const int _n0, const int _n1, const int _n2, 
      const int _s0, const int _s1, const int _s2, const int _coffs=1, const int _dev=0): 
      Ctensor3_view(_arr,_n0,_n1,_n2,_s0,_s1,_s2,_coffs,_dev){
      assert(n2==n1);
      assert(n1%2==1);
      l=(n1-1)/2;
      ar=arr+l*s1+l*s2;
      ac=arrc+l*s1+l*s2;
    }

    SO3Fpart3_view(float* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _coffs=1, 
    const int _dev=0):
      Ctensor3_view(_arr,_dims,_strides,_coffs,_dev){
      assert(n2==n1);
      assert(n1%2==1);
      l=(n1-1)/2;
      ar=arr+l*s1+l*s2;
      ac=arrc+l*s1+l*s2;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    int getl() const{
      return l;
    }

    complex<float> operator()(const int i0, const int m0, const int m1){
      int t=s0*i0+s1*m0+s2*m1;
      return complex<float>(ar[t],ac[t]);
    }

    void set(const int i0, const int m0, const int m1, complex<float> x){
      int t=s0*i0+s1*m0+s2*m1;
      ar[t]=std::real(x);
      ac[t]=std::imag(x);
    }

    void inc(const int i0, const int m0, const int m1, complex<float> x){
      int t=s0*i0+s1*m0+s2*m1;
      ar[t]+=std::real(x);
      ac[t]+=std::imag(x);
    }


  public: // ---- Other views -------------------------------------------------------------------------------

    
    SO3Fpart2_view slice0(const int i) const{
      return SO3Fpart2_view(arr+i*s0,arrc+i*s0,n1,n2,s1,s2);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    SO3Fpart3_view& flip(){
      arr=arr+(n1-1)*s1+(n2-1)*s2;
      arrc=arrc+(n1-1)*s1+(n2-1)*s2;
      ar=arr-l*s1-l*s2;
      ac=arrc-l*s1-l*s2;
      s1=-s1;
      s2=-s2;
      return *this;
    }

  

  };





}

#endif 
