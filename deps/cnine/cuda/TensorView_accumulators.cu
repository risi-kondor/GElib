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

#ifndef _Ctensor1view_add_cu
#define _Ctensor1view_add_cu
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

//#include "Cmaps.hpp"
#include "Ctensor2_view.hpp"
#include "Rmask1.hpp"
//#include "AccumulateCmap.hpp"



__global__ void accumulator_kernel(float* rarr, float* xarr, const int* ptr, const float* tbl, 
  const int rstride, const int xstride, const int n0){
  
  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);
  const int t=threadIdx.x;

  const int lines=n0/32;
  const int tail=n0-lines*32;
  const int offs=ptr[blockIdx.x]+2;
  const int n=tbl[offs-1];
  float* target=rarr+((int)tbl[offs-2])*rstride;

  for(int j=0; j<lines; j++)
    shared[j*32+t]=target[j*32+t];
  if(t<tail)
    shared[lines*32+t]=target[lines*32+t];
  __syncthreads();
    
  for(int i=0; i<n; i++){
    float* src=xarr+((int)tbl[offs+2*i])*xstride;
    float c=tbl[offs+2*i+1];

    for(int j=0; j<lines; j++)
      shared[j*32+t]+=c*src[j*32+t];
    if(t<tail)
      shared[lines*32+t]+=c*src[lines*32+t];
  
    __syncthreads();
  }

  for(int j=0; j<lines; j++)
    target[j*32+t]=shared[j*32+t];
  if(t<tail)
    target[lines*32+t]=shared[lines*32+t];
  
}


namespace cnine{

  void Ctensor2view_accumulator_cu(const Ctensor2_view& r, const Ctensor2_view& x, const Rmask1& mask, const cudaStream_t& stream){

    mask.prepare();

    assert(r.dev==1);
    assert(x.dev==1);
    assert(r.arrc==r.arr+1);
    assert(x.arrc==x.arr+1);
    assert(r.s1==2);
    assert(x.s1==2);
    assert(x.n1==r.n1);

    int B=mask.lists.size();
    int nlines=cnine::roundup(x.n1*2,32)/32;
    assert(nlines<=384);

    accumulator_kernel<<<B,32,nlines*128,stream>>>
      (r.arr,x.arr,mask.ptrg,mask.arrg,r.s0,x.s0,x.n1);
    
  }

}

#endif 
