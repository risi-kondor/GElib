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


#ifndef _GivensSelectSlicesFn
#define _GivensSelectSlicesFn

#include "RtensorView.hpp"
#include "../../tensor_views/Itensor1_view.hpp"
#include "../../tensor_views/Itensor2_view.hpp"


namespace cnine{

  #ifdef _WITH_CUDA
  extern void GivensSomeSlices_cu(const Rtensor2_view& x, const Itensor2_view& indices, const Rtensor2view& coeffs, const cudaStream_t& stream);
  #endif 


  class GivensSomeSlicesFn{
  public:

    void operator()(const Rtensor2_view& x, const Itensor2_view& indices, const Rtensor2_view& coeffs){
      CNINE_DEVICE_EQ(x,indices);
      CNINE_DEVICE_EQ(x,coeffs);
      assert(indices.n0==coeffs.n0);
      assert(indices.n1==2);
      assert(coeffs.n1==2);
      
      if(x.dev==0){
	int n=coeffs.n0;
	int s0=x.s0;
	int s1=x.s1;
	for(int i=0; i<n; i++){
	  int ix0=indices(i,0);
	  int ix1=indices(i,1);
	  float c0=coeffs(i,0);
	  float c1=coeffs(i,1);
	  for(int i1=0; i1<x.n1; i1++){
	    float t0=x.arr[ix0*s0+i1*s1];
	    float t1=x.arr[ix1*s0+i1*s1];
	    x.arr[ix0*s0+i1*s1]=c0*t0+c1*t1;
	    x.arr[ix1*s0+i1*s1]=c0*t1-c1*t0;
	  }
	}
      }
      if(x.dev==1){
	CUDA_STREAM(GivensSomeSlices_cu(x,indices,coeffs,stream));
      }
    }

    void operator()(const Rtensor3_view& x, const Itensor2_view& indices, const Rtensor2_view& coeffs){
      CNINE_DEVICE_EQ(x,indices);
      CNINE_DEVICE_EQ(x,coeffs);
      assert(indices.n0==coeffs.n0);
      assert(indices.n1==2);
      assert(coeffs.n1==2);
      
      if(x.dev==0){
	int n=coeffs.n0;
	int s0=x.s0;
	int s1=x.s1;
	int s2=x.s2;
	for(int i=0; i<n; i++){
	  int ix0=indices(i,0);
	  int ix1=indices(i,1);
	  float c0=coeffs(i,0);
	  float c1=coeffs(i,1);
	  for(int i1=0; i1<x.n1; i1++)
	    for(int i2=0; i2<x.n2; i2++){
	      float t0=x.arr[ix0*s0+i1*s1+i2*s2];
	      float t1=x.arr[ix1*s0+i1*s1+i2*s2];
	      x.arr[ix0*s0+i1*s1+i2*s2]=c0*t0+c1*t1;
	      x.arr[ix1*s0+i1*s1+i2*s2]=-(c0*t1-c1*t0); // hack!!!
	    }
	}
      }
      if(x.dev==1){
	CUDA_STREAM(GivensSomeSlices_cu(x,indices,coeffs,stream));
      }
    }


  };

}


#endif
