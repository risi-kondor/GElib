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


#ifndef _Ctensor_mprodFn
#define _Ctensor_mprodFn

#include "CtensorB_accessor.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{

  class Ctensor_add_mprod_AA{
  public:

    void operator()(Ctensor2_view& r, const Ctensor2_view& x, const Ctensor2_view& y){
      int I=r.n0;
      int J=r.n1;
      int K=x.n1;

      assert(x.n0==I);
      assert(y.n1==J);
      assert(y.n0==K);

      for(int i=0; i<I; i++)
	for(int j=0; j<J; j++){
	  decltype(r(0,0)) t=0;
	  for(int k=0; k<K; k++)
	    t+=x(i,k)*y(k,j);
	  r.inc(i,j,t);
	}
    }

  };

}


#endif 
