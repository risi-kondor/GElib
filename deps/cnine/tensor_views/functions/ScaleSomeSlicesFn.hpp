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


#ifndef _ScaleSelectSlicesFn
#define _ScaleSelectSlicesFn

#include "RtensorView.hpp"
#include "../../tensor_views/Itensor1_view.hpp"

namespace cnine{

  #ifdef _WITH_CUDA
  extern void ScaleSomeSlices_cu(const Rtensor2_view& x, const Itensor1_view& indices, const Rtensor1view& coeffs, const cudaStream_t& stream);
  #endif 


  class ScaleSomeSlicesFn{
  public:

    void operator()(const Rtensor2_view& x, const Itensor1_view& indices, const Rtensor1_view& coeffs){
      CNINE_DEVICE_EQ(x,indices);
      CNINE_DEVICE_EQ(x,coeffs);
      assert(indices.n0==coeffs.n0);
      
      if(x.dev==0){
	int n=coeffs.n0;
	int s0=x.s0;
	int s1=x.s1;
	for(int i=0; i<n; i++){
	  int ix=indices(i);
	  float c=coeffs(i);
	  for(int i1=0; i1<x.n1; i1++)
	    x.arr[ix*s0+i1*s1]*=c;
	}
      }
      if(x.dev==1){
	CUDA_STREAM(ScaleSomeSlices_cu(x,indices,coeffs,stream));
      }
    }

    void operator()(const Rtensor3_view& x, const Itensor1_view& indices, const Rtensor1_view& coeffs){
      CNINE_DEVICE_EQ(x,indices);
      CNINE_DEVICE_EQ(x,coeffs);
      assert(indices.n0==coeffs.n0);
      
      if(x.dev==0){
	int n=coeffs.n0;
	int s0=x.s0;
	int s1=x.s1;
	int s2=x.s2;
	for(int i=0; i<n; i++){
	  int ix=indices(i);
	  float c=coeffs(i);
	  for(int i1=0; i1<x.n1; i1++)
	    for(int i2=0; i2<x.n2; i2++)
	      x.arr[ix*s0+i1*s1+i2*s2]*=c;
	}
      }
      if(x.dev==1){
	CUDA_STREAM(ScaleSomeSlices_cu(x,indices,coeffs,stream));
      }
    }

  };

}


#endif
