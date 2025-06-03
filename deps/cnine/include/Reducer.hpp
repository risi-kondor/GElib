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


#ifndef _CnineReducer
#define _CnineReducer

#include "Cnine_base.hpp"

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{

  class ReduceAddDestroy_cfloat{
  public:

    ReduceAddDestroy_cfloat(float* dest, float* destc, float* source, float* sourcec, 
      const int cellstride, const int n){
      if(n==0) return;

      const float alpha=1;
      int s=0;
      int e=1;
      while(e<n){e*=2;s++;}
      e/=2; 

      CUBLAS_SAFE(cublasSaxpy(cnine_cublas, (n-e)*cellstride, &alpha, source+e*cellstride, 1, source, 1));
      CUBLAS_SAFE(cublasSaxpy(cnine_cublas, (n-e)*cellstride, &alpha, sourcec+e*cellstride, 1, sourcec, 1));

      for(int i=0; i<s-1; i++){
	e/=2;
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, e*cellstride, &alpha, source+e*cellstride, 1, source, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, e*cellstride, &alpha, sourcec+e*cellstride, 1, sourcec, 1));
      }

      CUBLAS_SAFE(cublasSaxpy(cnine_cublas, cellstride, &alpha, source, 1, dest, 1));
      CUBLAS_SAFE(cublasSaxpy(cnine_cublas, cellstride, &alpha, sourcec, 1, destc, 1));

    }

  };

  
  class ReduceAdd_cfloat{
  public:

    ReduceAdd_cfloat(float* dest, float* destc, const float* source, const float* sourcec, 
      const int cellstride, const int n){
      if(n==0) return;

      const float alpha=1;
      int s=0;
      int e=1;
      while(e<n){e*=2;s++;}
      e/=2; 

      float* buf=nullptr;
      float* bufc;
      CUDA_SAFE(cudaMalloc((void **)&buf, 2*e*cellstride*sizeof(float)));
      bufc=buf+e*cellstride;

      CUDA_SAFE(cudaMemcpy(buf,source,e*cellstride*sizeof(float),cudaMemcpyDeviceToDevice));  
      CUDA_SAFE(cudaMemcpy(bufc,sourcec,e*cellstride*sizeof(float),cudaMemcpyDeviceToDevice));  

      CUBLAS_SAFE(cublasSaxpy(cnine_cublas, (n-e)*cellstride, &alpha, source+e*cellstride, 1, buf, 1));
      CUBLAS_SAFE(cublasSaxpy(cnine_cublas, (n-e)*cellstride, &alpha, sourcec+e*cellstride, 1, bufc, 1));

      for(int i=0; i<s-1; i++){
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, e*cellstride, &alpha, buf+e*cellstride, 1, buf, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, e*cellstride, &alpha, buf+e*cellstride, 1, bufc, 1));
	e/=2;
      }

      CUBLAS_SAFE(cublasSaxpy(cnine_cublas, cellstride, &alpha, buf, 1, dest, 1));
      CUBLAS_SAFE(cublasSaxpy(cnine_cublas, cellstride, &alpha, bufc, 1, destc, 1));

    }

  };

  

}

#endif
