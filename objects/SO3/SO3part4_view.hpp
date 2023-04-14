
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3part4_view
#define _SO3part4_view

#include "Ctensor4_view.hpp"
#include "SO3part3_view.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;



namespace GElib{

  class SO3part4_view: public cnine::Ctensor4_view{
  public:

    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::Gdims Gdims;
    typedef cnine::Gstrides Gstrides;

    int l;
    float* ar;
    float* ac;

  public:


    SO3part4_view(){}

    SO3part4_view(float* _arr, const int _n0, const int _n1, const int _n2, const int _n3, 
      const int _s0, const int _s1, const int _s2, const int _s3,  const int _coffs=1, const int _dev=0): 
      Ctensor4_view(_arr,_n0,_n1,_n2,_n3,_s0,_s1,_s2,_s3,_coffs,_dev){
      assert(n2%2==1);
      l=(n2-1)/2;
      ar=arr+l*s2;
      ac=arrc+l*s2;
    }

    SO3part4_view(float* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _coffs=1, const int _dev=0):
      Ctensor4_view(_arr,_dims,_strides,_coffs,_dev){
      assert(n2%2==1);
      l=(n2-1)/2;
      ar=arr+l*s2;
      ac=arrc+l*s2;
    }

    SO3part4_view(const Ctensor4_view& x):
      Ctensor4_view(x){
      assert(n2%2==1);
      l=(n2-1)/2;
      ar=arr+l*s2;
      ac=arrc+l*s2;
    }
      

  public: // ---- Access ------------------------------------------------------------------------------------


    int getl() const{
      return l;
    }

    complex<float> operator()(const int i0, const int i1, const int m, const int i3){
      int t=s0*i0+s1*i1+s2*m+s3*i3;
      return complex<float>(ar[t],ac[t]);
    }

    void set(const int i0, const int i1, const int m, const int i3, complex<float> x){
      int t=s0*i0+s1*i1+s2*m+s3*i3;
      ar[t]=std::real(x);
      ac[t]=std::imag(x);
    }

    void inc(const int i0, const int i1, const int m, const int i3, complex<float> x){
      int t=s0*i0+s1*i1+s2*m+s3*i3;
      ar[t]+=std::real(x);
      ac[t]+=std::imag(x);
    }


  public: // ---- Other views -------------------------------------------------------------------------------

    
    SO3part3_view slice0(const int i) const{
      return SO3part3_view(arr+i*s0,arrc+i*s0,n1,n2,n3,s1,s2,s3,dev);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<n0; i++){
	oss<<indent<<"b="<<i<<":"<<endl;
	oss<<slice0(i).str(indent)<<endl;
      }
      return oss.str();
    }


  };





}

#endif 
