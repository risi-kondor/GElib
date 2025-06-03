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


#ifndef _CnineRtensorReduceFn
#define _CnineRtensorReduceFn

#include "RtensorView.hpp"

#ifdef _WITH_CUDA
extern void InplaceReduce1_cu(const Rtensor3_view& x, const cudaStream_t& stream);
#endif 

namespace cnine{


  class Rtensor3_view_reduce1_Fn{
  public:

    void operator()(const Rtensor3_view& x){
      
      if(x.dev==0){
	for(int i0=0; i0<x.n0; i0++){
	  for(int i2=0; i2<x.n2; i2++){
	    float t=x.arr[s0*i0+s2*i2];
	    for(int i1=1; i1<x.n1; i1++)
	      t+=x.arr[x.s0*i0+x.s1*i1+x.s2*i2];
	    x.arr[x.s0*i0+x.s2*i2]=t;
	  }
	}
      }
      
      if(x.dev==1){
	CUDA_STREAM(InplaceReduce1_cu(x,stream));
      }

    }
    
  };

}
