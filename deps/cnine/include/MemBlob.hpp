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


#ifndef _CnineMemBlob
#define _CnineMemBlob

#include "Cnine_base.hpp"
#include "CnineCallStack.hpp"
#include "MemoryManager.hpp"
#include "fnlog.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{

  extern CallStack call_stack;
  extern thread_local MemoryManager* vram_manager;
  extern CnineLog cnine_log;


  template<typename TYPE>
  class MemBlob{
  public:

    typedef std::size_t size_t;

    TYPE* arr;
    int dev=0;
    bool is_view=false;
    const MemoryManager* manager=nullptr;

    ~MemBlob(){
      if(is_view) return;
      if(manager){
	manager->free(static_cast<void*>(arr));
	return;
      }
      BLOB_DEBUG("Delete blob.");
      if(dev==0 && arr) {delete[] arr;}
      if(dev==1 && arr) {CUDA_SAFE(cudaFree(arr));}
    }

  public: // ---- Constructors ------------------------------------------------------------------------------


    MemBlob(size_t _memsize, const int _dev=0):
      dev(_dev){
      if(_memsize<1) _memsize=1;
      BLOB_DEBUG("New blob of size "+to_string(_memsize)+" on device "+to_string(_dev)+".");

      if(vram_manager && _dev>0){
	//fnlog timer("MemBlob managed");
	manager=vram_manager;
	arr=static_cast<TYPE*>(manager->malloc(_memsize*sizeof(TYPE)));
	return;
      }

      //fnlog timer("MemBlob not managed");
      CPUCODE(arr=new TYPE[_memsize];);
      GPUCODE(CUDA_SAFE(cudaMalloc((void **)&arr, _memsize*sizeof(TYPE))););
    }

    MemBlob(const MemoryManager& _manager, size_t _memsize, const int _dev=0):
      dev(_dev),
      manager(&_manager){
      //fnlog timer("MemBlob explicitly managed");
      arr=static_cast<TYPE*>(manager->malloc(_memsize*sizeof(TYPE)));
    }


  public: // ---- Views --------------------------------------------------------------------------------------


    // just for taking views of ATen tensors 
    MemBlob(TYPE* _arr, const int _dev=0):
      arr(_arr), 
      dev(_dev), 
      is_view(true){}

    MemBlob(int _dev, TYPE* _arr):
      arr(_arr), 
      dev(_dev),
      is_view(true){}


  };

}

#endif 
