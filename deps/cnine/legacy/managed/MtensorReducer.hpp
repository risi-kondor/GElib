/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _MtensorReducer
#define _MtensorReducer

#include "CtensorBpack.hpp"


namespace cnine{

  template<typename TYPE>
  class MtensorReducer: public CtensorBpack{
  public:

    Tensor<TYPE>& target;

    MtensorReducer(const int _N, Tensor<TYPE>& _target):
      CtensorBpack(_N,_target.dims,_target.nbu,1), 
      target(_target){

      N=_N;
      int cst=target.cst;
      int memsize=target.memsize;
      CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*N*sizeof(float)));
      CUDA_SAFE(cudaMemset(arrg,0,N*memsize*sizeof(float)));
      arrgc=arrg+target.cst;

      float* arr[N]; 
      float* arrc[N]; 
      for(int i=0; i<N; i++){
	arr[i]=arrg+i*memsize;
	arrc[i]=arrgc+i*memsize;
      }
      
      //CUDA_SAFE(cudaMalloc((void ***)&parr, N*sizeof(float*)));
      //CUDA_SAFE(cudaMalloc((void ***)&parrc, N*sizeof(float*)));
      CUDA_SAFE(cudaMemcpy(parr,arr,N*sizeof(float*),cudaMemcpyHostToDevice));  
      CUDA_SAFE(cudaMemcpy(parrc,arrc,N*sizeof(float*),cudaMemcpyHostToDevice));  

      parr_valid=true;
    }


    ~CtensorBreducer(){
#ifdef _WITH_CUDA
	target.to_device(1);
	cudaStream_t stream;
	cudaDeviceSynchronize();
	CUDA_SAFE(cudaStreamCreate(&stream));
	sum_into_cu(target,stream);
	CUDA_SAFE(cudaStreamSynchronize(stream));
	CUDA_SAFE(cudaStreamDestroy(stream));
	cudaDeviceSynchronize();
#else
	NOCUDA_ERROR;
#endif
    }


  public:


  };

}

#endif
