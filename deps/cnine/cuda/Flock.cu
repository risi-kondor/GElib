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

#ifndef _Flock_cu
#define _Flock_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include "Flock.hpp"




  // ---- COPY -----------------------------------------------------------------------------------------------


__global__ void Flock_copy_kernel(float** arr_array, float** x_arr_array, int nwarps){
  
  const int b=blockIdx.x;
  const int t=threadIdx.x;
    
  float* ptr=arr_array[b];
  float* xptr=x_arr_array[b];
    
  for(int i=0; i<nwarps; i++)
      ptr[i*32+t]=xptr[i*32+t];
  
}


namespace cnine{
  template<typename OBJ>
  void Flock<OBJ>::copy_cu(const Flock& x, const cudaStream_t& stream){
    Flock_copy_kernel<<<N,32,0,stream>>>
      (arr_array,x.arr_array,memsize/32);
  }
}

  // ---- ADD TO SINGLETON ----------------------------------------------------------------------------------


  __global__ void Flock_add_to_singleton_kernel(float* dest, float** arr_array, int nwarps){
    
    const int b=blockIdx.x;
    const int t=threadIdx.x;
    
    float* ptr=arr_array[b];
    
    for(int i=0; i<nwarps; i++)
      dest[i*32+t]+=ptr[i*32+t];
    
  }


  // ---- ADD ------------------------------------------------------------------------------------------------


  __global__ void Flock_add_kernel(float** arr_array, float** x_arr_array, int nwarps){

    const int b=blockIdx.x;
    const int t=threadIdx.x;

    float* ptr=arr_array[b];
    float* xptr=x_arr_array[b];

    for(int i=0; i<nwarps; i++)
      ptr[i*32+t]+=xptr[i*32+t];
  }


namespace cnine{
  template<typename OBJ>
  void Flock<OBJ>::add_cu(const Flock& x, const cudaStream_t& stream){
    Flock_add_kernel<<<N,32,0,stream>>>
      (arr_array,x.arr_array,memsize/32); 
  }
}
  
  // ---- SUM_INTO -------------------------------------------------------------------------------------------


  __global__ void Flock_reduce_kernel(float** arr_array, const int nwarps, const int d){
    
    const int b=blockIdx.x;
    const int t=threadIdx.x;
    
    float* ptr=arr_array[b];
    float* xptr=arr_array[b+d];
    for(int i=0; i<nwarps; i++)
      ptr[i*32+t]+=xptr[i*32+t];
    
  }


namespace cnine{
  template<typename OBJ>
  void Flock<OBJ>::sum_into_cu(const OBJ& R, const cudaStream_t& stream){

    int d=1;
    while(d<N) d*=2;
    d/=2;

    Flock_reduce_kernel<<<N-d,32,0,stream>>>
      (arr_array,memsize/32,d);
    d/=2;

    while(d>0){
      Flock_reduce_kernel<<<d,32,0,stream>>>
	(arr_array,memsize/32,d);
      d/=2;
    }

    Flock_add_to_singleton_kernel<<<1,32,0,stream>>>
      (R.arrg,arr_array,memsize/32);

  }
}


#endif 

