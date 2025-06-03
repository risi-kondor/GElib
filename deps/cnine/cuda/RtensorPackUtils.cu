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

#ifndef _RtensorUtils_cu
#define _RtensorUtils_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

#include "Cnine_base.hpp"
#include "RtensorPack.hpp"
#include "CUDAhelpers.hpp"


// ---- ReLU ------------------------------------------------------------------------------------------------


__global__ void RtensorPack_add_ReLU_kernel(float* rarr, const float* xarr, const float alpha){
  float v=xarr[blockIdx.x*32+threadIdx.x];
  if(v>0) rarr[blockIdx.x*32+threadIdx.x]=v;
  else rarr[blockIdx.x*32+threadIdx.x]=alpha*v;
}

__global__ void RtensorPack_add_ReLU_back_kernel(float* rarr, const float* garr, const float* xarr, const float alpha){
  float v=garr[blockIdx.x*32+threadIdx.x];
  if(xarr[blockIdx.x*32+threadIdx.x]>0) 
    rarr[blockIdx.x*32+threadIdx.x]=v;
  else 
    rarr[blockIdx.x*32+threadIdx.x]=alpha*v;
}


// ----------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------


namespace cnine{

  void RtensorPack_add_ReLU_cu(RtensorPack& r, const RtensorPack& x, const float alpha, const cudaStream_t& stream){
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(r.tail==x.tail);
    if(r.tail>=32) RtensorPack_add_ReLU_kernel<<<r.tail/32,32,0,stream>>>(r.arrg,x.arrg,alpha);
    if(r.tail%32>0) RtensorPack_add_ReLU_kernel<<<1,r.tail%32,0,stream>>>(r.arrg+(r.tail/32)*32,x.arrg+(r.tail/32)*32,alpha);
  }

  void RtensorPack_add_ReLU_back_cu(RtensorPack& r, const RtensorPack& g, const RtensorPack& x, const float alpha, const cudaStream_t& stream){
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(g.dev==1);
    CNINE_ASSRT(r.tail==x.tail);
    CNINE_ASSRT(r.tail==g.tail);
    if(r.tail>=32) RtensorPack_add_ReLU_back_kernel<<<r.tail/32,32,0,stream>>>(r.arrg,g.arrg,x.arrg,alpha);
    if(r.tail%32>0) RtensorPack_add_ReLU_back_kernel<<<1,r.tail%32,0,stream>>>(r.arrg+(r.tail/32)*32,g.arrg+(r.tail/32)*32,x.arrg+(r.tail/32)*32,alpha);
  }


}


#endif 
