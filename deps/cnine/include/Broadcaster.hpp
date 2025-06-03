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


#ifndef _CnineBroadcaster
#define _CnineBroadcaster

#include "Cnine_base.hpp"

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{

  class BroadcastCopy_cfloat{
  public:
    
    BroadcastCopy_cfloat(float* dest, float* destc, const float* source, const float* sourcec, 
      const int cellstride, const int n){
      if(n==0) return;

      int s=0;
      int e=1;
      while(e<=n){e*=2;s++;}

      CUDA_SAFE(cudaMemcpy(dest,source,cellstride*sizeof(float),cudaMemcpyDeviceToDevice));  
      CUDA_SAFE(cudaMemcpy(destc,sourcec,cellstride*sizeof(float),cudaMemcpyDeviceToDevice));  

      e=1; 
      for(int i=0; i<s-1; i++){
	CUDA_SAFE(cudaMemcpy(dest+e*cellstride,dest,e*cellstride*sizeof(float),cudaMemcpyDeviceToDevice));  
	CUDA_SAFE(cudaMemcpy(destc+e*cellstride,destc,e*cellstride*sizeof(float),cudaMemcpyDeviceToDevice));  
	e*=2;
      }

      CUDA_SAFE(cudaMemcpy(dest+e*cellstride,dest,(n-e)*cellstride*sizeof(float),cudaMemcpyDeviceToDevice));  
      CUDA_SAFE(cudaMemcpy(destc+e*cellstride,destc,(n-e)*cellstride*sizeof(float),cudaMemcpyDeviceToDevice));  

    }
    
  };


  class BroadcastAdd_cfloat{
  public:
    
    BroadcastAdd_cfloat(float* dest, float* destc, const float* source, const float* sourcec, 
      const int cellstride, const int n){
      if(n==0) return;

      const float alpha=1;
      int s=0;
      int e=1;
      while(e<=n){e*=2;s++;}

      CUBLAS_SAFE(cublasSaxpy(cnine_cublas, cellstride, &alpha, source, 1, dest, 1));
      CUBLAS_SAFE(cublasSaxpy(cnine_cublas, cellstride, &alpha, sourcec, 1, destc, 1));

      e=1; 
      for(int i=0; i<s-1; i++){
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, e*cellstride, &alpha, dest, 1, dest+e*cellstride, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, e*cellstride, &alpha, destc, 1, destc+e*cellstride, 1));
	e*=2;
      }

      CUBLAS_SAFE(cublasSaxpy(cnine_cublas, (n-e)*cellstride, &alpha, dest, 1, dest+e*cellstride, 1));
      CUBLAS_SAFE(cublasSaxpy(cnine_cublas, (n-e)*cellstride, &alpha, destc, 1, destc+e*cellstride, 1));

      }
    
  };

}

#endif
