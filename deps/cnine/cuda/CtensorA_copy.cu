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

#ifndef _CtensorA_copy_cu
#define _CtensorA_copy_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

#include "CtensorArrayA.hpp"
#include "CellwiseUnaryCmap.hpp"


template<typename CMAP>
__global__ void CtensorA_copy_kernel(float* rarr, float* rarrc, float* xarr, float* xarrc, 
  const int rstride, const int xstride, const CMAP& cmap){
  
  auto T=cmap(blockIdx.x,blockIdx.y);
  const int rix=thrust::get<0>(T);
  const int xix=thrust::get<1>(T);
  const int t=threadIdx.x;

  rarr[rix*rstride+t]=xarr[xix*xstride+t];
  rarrc[rix*rstride+t]=xarrc[xix*xstride+t];
}


template<typename CMAP>
__global__ void CtensorA_add_copy_kernel(float* rarr, float* rarrc, float* xarr, float* xarrc, 
  const int rstride, const int xstride, const CMAP& cmap){
  
  auto T=cmap(blockIdx.x,blockIdx.y);
  const int rix=thrust::get<0>(T);
  const int xix=thrust::get<1>(T);
  const int t=threadIdx.x;

  rarr[rix*rstride+t]+=xarr[xix*xstride+t];
  rarrc[rix*rstride+t]+=xarrc[xix*xstride+t];
}


template<typename CMAP>
__global__ void CtensorA_copy_accumulator_kernel(float* rarr, float* rarrc, float* xarr, float* xarrc, 
  const int rstride, const int xstride, const CMAP& cmap){

  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);
 
  const int n_accum=cmap.n_accum(blockIdx.x);
  const int rix=cmap.target(blockIdx.x);
  const int lst_ptr=cmap.lst_ptr(blockIdx.x);
  const int t=threadIdx.x;

  shared[t]=rarr[rix*rstride+t];
  shared[t+rstride]=rarrc[rix*rstride+t];
  
  for(int i=0; i<n_accum; i++){
    const int xix=cmap.source(blockIdx.x,lst_ptr);
    shared[t]+=xarr[xix*xstride+t];
    shared[t+rstride]+=xarrc[xix*xstride+t];
  }
  
  rarr[rix*rstride+t]=shared[t];
  rarrc[rix*rstride+t]=shared[t+rstride];
  
}


namespace cnine{


  template<typename CMAP>
  void CtensorA_copy_cu(const CMAP& cmap, CtensorArrayA& r, const CtensorArrayA& x, const cudaStream_t& stream){

    CtensorA_copy_kernel<<<cmap.blockdims(),r.asize,0,stream>>>
      (r.arrg,r.arrgc,x.arrg,x.arrgc,r.cellstride,x.cellstride,cmap);
    
  }

  template<typename CMAP>
  void CtensorA_add_copy_cu(const CMAP& cmap, CtensorArrayA& r, const CtensorArrayA& x, const cudaStream_t& stream){

    CtensorA_add_copy_kernel<<<cmap.blockdims(),r.asize,0,stream>>>
      (r.arrg,r.arrgc,x.arrg,x.arrgc,r.cellstride,x.cellstride,cmap);
    
  }

  template<typename CMAP>
  void CtensorA_copy_accumulator_cu(const CMAP& cmap, CtensorArrayA& r, const CtensorArrayA& x, const cudaStream_t& stream){

    CtensorA_copy_accumulator_kernel<<<cmap.blockdims(),r.asize,0,stream>>>
      (r.arrg,r.arrgc,x.arrg,x.arrgc,r.cellstride,x.cellstride,cmap);
    
  }


  template void CtensorA_copy_cu(const cnine::CellwiseUnaryCmap& map, 
    CtensorArrayA&, const CtensorArrayA&, const cudaStream_t&);

  template void CtensorA_add_copy_cu(const cnine::CellwiseUnaryCmap& map, 
    CtensorArrayA&, const CtensorArrayA&, const cudaStream_t&);


}

#endif 


  /*
  template<typename CMAP, typename = typename std::enable_if<std::is_base_of<DirectCmap, CMAP>::value, CMAP>::type>
  void CtensorA_copy_cu(const CMAP& cmap, CtensorArrayA& r, const CtensorArrayA& x, const cudaStream_t& stream){

    CtensorA_copy_kernel<<<cmap.blockdims(),r.asize,0,stream>>>
      (r.arrg,r.arrgc,x.arrg,x.arrgc,r.cellstride,x.cellstride,cmap);
    
  }

  template<typename CMAP, typename = typename std::enable_if<std::is_base_of<AccumulatorCmap, CMAP>::value, CMAP>::type>
  void CtensorA_copy_cu(const CMAP& cmap, CtensorArrayA& r, const CtensorArrayA& x, const cudaStream_t& stream){

    CtensorA_copy_accumulator_kernel<<<cmap.blockdims(),r.asize,0,stream>>>
      (r.arrg,r.arrgc,x.arrg,x.arrgc,r.cellstride,x.cellstride,cmap);
    
  }
  */

  /*
  template<typename CMAP, typename = typename std::enable_if<std::is_base_of<DirectCmap, CMAP>::value, CMAP>::type>
  void CtensorA_copy_cu(const CMAP& cmap, CtensorArrayA& r, const CtensorArrayA& x, const cudaStream_t& stream){

    int cmap_type=cmap.cmap_type();

    if(cmap_type==0){
      CtensorA_copy_kernel<<<cmap.blockdims(),r.asize,0,stream>>>
	(r.arrg,r.arrgc,x.arrg,x.arrgc,r.cellstride,x.cellstride,cmap);
    }

    if(cmap_type==2){
      CtensorA_copy_accumulator_kernel<<<cmap.blockdims(),r.asize,r.cellstride*sizeof(float),stream>>>
	(r.arrg,r.arrgc,x.arrg,x.arrgc,r.cellstride,x.cellstride,cmap);
    }


  }
  */


