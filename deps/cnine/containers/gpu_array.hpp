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

#ifndef _gpu_array
#define _gpu_array

#include "Cnine_base.hpp"
#include "int_array.hpp"


namespace cnine{

  template<typename TYPE>
  class gpu_array{
  public:

    typedef std::size_t size_t;

    TYPE* arr;
    size_t _size;
    int dev=0;

    ~gpu_array(){
      CUDA_SAFE(cudaFree(arr));
    }


  public: //---- Constructors -------------------------------------


    gpu_array(const int n, const int _dev=1){
      _size=std::max(1,n);
      CUDA_SAFE(cudaMalloc((void **)&arr,_size*sizeof(TYPE)));
    }

    gpu_array(const vector<TYPE>& x, const int _dev=1):
      gpu_array(x.size(),_dev){
      CUDA_SAFE(cudaMemcpy(arr,&const_cast<vector<TYPE> >(x)[0],_size*sizeof(TYPE),cudaMemcpyHostToDevice)); 
    }

    gpu_array(const int_array& x, const int _dev=1):
      gpu_array(x.memsize,_dev){
      CUDA_SAFE(cudaMemcpy(arr,x.arr,_size*sizeof(TYPE),cudaMemcpyHostToDevice)); 
    }


  public: //---- Access -------------------------------------------


    operator TYPE*(){
      return arr;
    }

  };

}
