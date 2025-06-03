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


#ifndef _BasicCtensorProducts
#define _BasicCtensorProducts

#include "Cnine_base.hpp"
#include "Gstrides.hpp"

#include "Ctensor2_view.hpp"

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 

namespace cnine{


  #ifdef _WITH_CUDA

  void BasicCproduct_4_cu(const float* xarr, const float* xarrc, const float* yarr, const float* yarrc, float* rarr, float* rarrc, 
    const int n0, const int n1, const int n2, const int n3, 
    const int xs0, const int xs1, const int xs2, const int xs3, 
    const int ys0, const int ys1, const int ys2, const int ys3, 
    const int rs0, const int rs1,  const int rs2, const int rs3,
    const cudaStream_t& stream);

  void BasicCproduct_2_1_cu(const float* xarr, const float* xarrc, const float* yarr, const float* yarrc, float* rarr, float* rarrc, 
    const int n0, const int n1, const int n2,  
    const int xs0, const int xs1, const int xs2, 
    const int ys0, const int ys1, const int ys2, 
    const int rs0, const int rs1,  
    const cudaStream_t& stream);

  #endif 



  // ---- no summation ----------------------------------------------------------------------------------------


  template<typename TYPE>
  void BasicCproduct_4(const TYPE* xarr, const TYPE* xarrc, const TYPE* yarr, const TYPE* yarrc, TYPE* rarr, TYPE* rarrc, 
    const int n0, const int n1, const int n2, const int n3, 
    const int xs0, const int xs1, const int xs2, const int xs3, 
    const int ys0, const int ys1, const int ys2, const int ys3, 
    const int rs0, const int rs1,  const int rs2, const int rs3, 
    const int dev=0){

    if(dev==0){
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    for(int i3=0; i3<n3; i3++){
	      complex<TYPE> t=
		complex<TYPE>(xarr[i0*xs0+i1*xs1+i2*xs2+i3*xs3],xarrc[i0*xs0+i1*xs1+i2*xs2+i3*xs3])*
		complex<TYPE>(yarr[i0*ys0+i1*ys1+i2*ys2+i3*ys3],yarrc[i0*ys0+i1*ys1+i2*ys2+i3*ys3]);
	      rarr[i0*rs0+i1*rs1+i2*rs2+i3*rs3]=std::real(t);
	      rarrc[i0*rs0+i1*rs1+i2*rs2+i3*rs3]=std::imag(t);
	    }
    }
    if(dev==1){
      CUDA_STREAM(BasicCproduct_4_cu(xarr,xarrc,yarr,yarrc,rarr,rarrc,
	  n0,n1,n2,n3,xs0,xs1,xs2,xs3,ys0,ys1,ys2,ys3,rs0,rs1,rs2,rs3,stream));
    }
  }


  // ---- one summed index -----------------------------------------------------------------------------------


  template<typename TYPE>
  void BasicCproduct_1_1(const TYPE* xarr, const TYPE* xarrc, const TYPE* yarr, const TYPE* yarrc, 
    TYPE* rarr, TYPE* rarrc, const int n0, const int n1, const int xs0, const int xs1, 
    const int ys0, const int ys1, const int rs0, const int dev=0){
    if(dev==0){
      for(int i0=0; i0<n0; i0++){
	complex<TYPE> t=0;
	for(int i1=0; i1<n1; i1++){
	  t+=complex<TYPE>(xarr[i0*xs0+i1*xs1],xarrc[i0*xs0+i1*xs1])*
	    complex<TYPE>(yarr[i0*ys0+i1*ys1],yarrc[i0*ys0+i1*ys1]);
	  rarr[i0*rs0]=std::real(t);
	  rarrc[i0*rs0]=std::imag(t);
	}
      }
    }
    if(dev==1){
      CNINE_CPUONLY();
    }
  }


  template<typename TYPE>
  void BasicCproduct_2_1(const TYPE* xarr, const TYPE* xarrc, const TYPE* yarr, const TYPE* yarrc, TYPE* rarr, TYPE* rarrc, 
    const int n0, const int n1, const int n2, 
    const int xs0, const int xs1, const int xs2, 
    const int ys0, const int ys1, const int ys2, 
    const int rs0, const int rs1,
    const int dev=0){

    if(dev==0){
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++){
	  complex<TYPE> t=0;
	  for(int i2=0; i2<n2; i2++){
	    t+=complex<TYPE>(xarr[i0*xs0+i1*xs1+i2*xs2],xarrc[i0*xs0+i1*xs1+i2*xs2])*
	      complex<TYPE>(yarr[i0*ys0+i1*ys1+i2*ys2],yarrc[i0*ys0+i1*ys1+i2*ys2]);
	    rarr[i0*rs0+i1*rs1]=std::real(t);
	    rarrc[i0*rs0+i1*rs1]=std::imag(t);
	  }
	}
    }
    if(dev==1){
      CUDA_STREAM(BasicCproduct_2_1_cu(xarr,xarrc,yarr,yarrc,rarr,rarrc,
	  n0,n1,n2,xs0,xs1,xs2,ys0,ys1,ys2,rs0,rs1,stream));
    }
  }




}

#endif 
