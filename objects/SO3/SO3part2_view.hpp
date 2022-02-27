
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
#include "Ctensor2_view.hpp"

//#include "SO3_CGbank.hpp"
//#include "SO3_SPHgen.hpp"
//#include "SO3element.hpp"
//#include "WignerMatrix.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;



namespace GElib{

  class SO3part2_view: public cnine::Ctensor2_view{
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


    SO3part2_view(){}

    //SO3part2_view(float* _arr, float* _arrc): 
    //arr(_arr), arrc(_arrc){}

    SO3part2_view(float* _arr, float* _arrc, const int _n0, const int _n1, const int _s0, const int _s1, const int _dev=0): 
      Ctensor2_view(_arr,_arrc,_n0,_n1,_s0,_s1,_dev){
      //cout<<n0<<n1<<s0<<s1<<endl;
      assert(n0%2==1);
      l=(n0-1)/2;
      ar=arr+l*s0;
      ac=arrc+l*s0;
      //cout<<"l="<<l<<endl;
      //cout<<"diff="<<ac-ar<<endl;
    }

    SO3part2_view(float* _arr, const int _n0, const int _n1, const int _s0, const int _s1, const int _coffs=1, const int _dev=0): 
      Ctensor2_view(_arr,_n0,_n1,_s0,_s1,_coffs,_dev){
      assert(n0%2==1);
      l=(n0-1)/2;
      ar=arr+l*s0;
      ac=arrc+l*s0;
      //cout<<"diff="<<ac-ar<<endl;
    }

    SO3part2_view(float* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _coffs=1, const int _dev=0):
      Ctensor2_view(_arr,_dims,_strides,_coffs,_dev){
      l=(n0-1)/2;
      ar=arr+l*s0;
      ac=arrc+l*s0;
      //cout<<"diff="<<ac-ar<<endl;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    int getl() const{
      return l;
    }

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


  public: // ---- Other views -------------------------------------------------------------------------------

    
    //SO3part2_view slice2(const int i){
    //return SO3part2_view(arr+i*s2,arrc+i*s2,n0,n1,s0,s1);
    //}
  
  public: // ---- I/O ----------------------------------------------------------------------------------------

    /*
    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<n0; i++){
	oss<<"[ ";
	for(int j=0
	oss<<slice(0).str(indent+"  ")<<endl;
      }
      return oss.str();
    }
    */


  };





}

#endif 
