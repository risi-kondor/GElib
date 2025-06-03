/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _AsyncGPUbuffer
#define _AsyncGPUbuffer

#include "Cnine_base.hpp"


namespace cnine{

  template<typename TYPE>
  class AsyncGPUbuffer{
  public:

    typedef std::size_t size_t;

    TYPE* arr=nullptr;
    TYPE* arrg=nullptr;
    size_t _size=0;
    int dev=1;

    ~AsyncGPUbuffer(){
      if(arr) CUDA_SAFE(cudaFreeHost(arr));
      if(arrg) CUDA_SAFE(cudaFree(arrg));
    }


  public: //---- Constructors -------------------------------------


    AsyncGPUbuffer(){}

    AsyncGPUbuffer(const int n, const int _dev=1){
      _size=std::max(1,n);
      arr=new TYPE[_size];
      CUDA_SAFE(cudaHostAlloc((void **)&arr,_size*sizeof(TYPE),cudaHostAllocDefault));
      CUDA_SAFE(cudaMalloc((void **)&arrg,_size*sizeof(TYPE)));
      CUDA_SAFE(cudaDeviceSynchronize());
    }

    void resize(const int n, const int _dev=1){
      if(dev==_dev && n==_size) return;
      if(arr) CUDA_SAFE(cudaFreeHost(arr));
      if(arrg) CUDA_SAFE(cudaFree(arrg));
      _size=std::max(1,n);
      CUDA_SAFE(cudaHostAlloc((void **)&arr,_size*sizeof(TYPE),cudaHostAllocDefault));
      CUDA_SAFE(cudaMalloc((void **)&arrg,_size*sizeof(TYPE)));
      CUDA_SAFE(cudaDeviceSynchronize());
    }

    void min_size(const int n, const int _dev=1){
      if(n<_size || _dev!=dev) resize(n,_dev);
    }

    void reset(const int n, const int _dev=1){
      resize(n,_dev);
    }


  public: //---- Access -------------------------------------------


    TYPE* operator()(const int i=0){
      return arrg+i;
    }

    
    template<typename MINIVEC>
    void push_minivec(const int i, const MINIVEC& x){
      CNINE_ASSRT(x.dev==0);
      CNINE_ASSRT(i+x._size<=_size);
      std::copy(x.arr,x.arr+x._size,arr+i);
    }

    template<typename TENSOR>
    void push(const int i, const TENSOR& x){
      CNINE_ASSRT(x.get_dev()==0);
      CNINE_ASSRT(i+x.asize()<=_size);
      std::copy(x.mem(),x.mem()+x.asize(),arr+i);
    }

    void push(const int i, const vector<TYPE>& x){
      //CNINE_ASSRT(x.get_dev()==0);
      CNINE_ASSRT(i+x.size()<=_size);
      std::copy(x.begin(),x.end(),arr+i);
    }

#ifdef _WITH_CUDA
    void sync(const cudaStream_t& stream){
      CUDA_SAFE(cudaMemcpyAsync(arrg,arr,_size*sizeof(TYPE),cudaMemcpyHostToDevice,stream));
    }
#endif 

  };

}

#endif 
