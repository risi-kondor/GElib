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

#ifndef _CtensorA_plus_cu
#define _CtensorA_plus_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

#include "CtensorArrayA.hpp"
#include "CellwiseBinaryCmap.hpp"
#include "BroadcastBinaryCmap.hpp"
#include "InnerCmap.hpp"
#include "OuterCmap.hpp"
#include "MVprodCmap.hpp"
#include "VMprodCmap.hpp"
//#include "convolve1_cmap.hpp"
#include "Convolve2Cmap.hpp"
#include "accumulate_cmap.hpp"


template<typename IMAP>
__global__ void CtensorA_plus_kernel(float* rarr, float* rarrc, float* xarr, float* xarrc, 
  float* yarr, float* yarrc, const int rstride, const int xstride, const int ystride, const IMAP map){
  
  auto T=map(blockIdx.x,blockIdx.y,blockIdx.z);
  const int rix=thrust::get<0>(T);
  const int xix=thrust::get<1>(T);
  const int yix=thrust::get<2>(T);
  const int t=threadIdx.x;

  rarr[rix*rstride+t]=xarr[xix*xstride+t]+yarr[yix*ystride+t];
  rarrc[rix*rstride+t]=xarrc[xix*xstride+t]+yarrc[yix*ystride+t];
}


template<typename IMAP>
__global__ void CtensorA_add_plus_kernel(float* rarr, float* rarrc, float* xarr, float* xarrc, 
  float* yarr, float* yarrc, const int rstride, const int xstride, const int ystride, const IMAP map){
  
  auto T=map(blockIdx.x,blockIdx.y,blockIdx.z);
  const int rix=thrust::get<0>(T);
  const int xix=thrust::get<1>(T);
  const int yix=thrust::get<2>(T);
  const int t=threadIdx.x;

  rarr[rix*rstride+t]+=xarr[xix*xstride+t]+yarr[yix*ystride+t];
  rarrc[rix*rstride+t]+=xarrc[xix*xstride+t]+yarrc[yix*ystride+t];
}


template<typename CMAP>
__global__ void CtensorA_plus_accumulator_kernel(float* rarr, float* rarrc, float* xarr, float* xarrc, 
  float* yarr, float* yarrc, const int rstride, const int xstride, const int ystride, const CMAP cmap){

  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);
 
  const int n_accum=cmap.n_accum(blockIdx.x);
  const int rix=cmap.target(blockIdx.x);
  const int lst_ptr=cmap.lst_ptr(blockIdx.x);
  const int t=threadIdx.x;

  shared[t]=rarr[rix*rstride+t];
  shared[t+rstride]=rarrc[rix*rstride+t];

  //if(t==0){
  //printf("block %d: %d %d %d\n",blockIdx.x,n_accum,rix,lst_ptr);
  //}

  for(int i=0; i<n_accum; i++){
    auto T=cmap.source(lst_ptr,blockIdx.x,i);
    const int xix=thrust::get<0>(T);
    const int yix=thrust::get<1>(T);
    //if(t==0) printf("%d %d %d\n",rix,xix,yix);
    shared[t]+=xarr[xix*xstride+t]+yarr[yix*ystride+t];
    shared[t+rstride]+=xarrc[xix*xstride+t]+yarrc[yix*ystride+t];
  }
  
  rarr[rix*rstride+t]=shared[t];
  rarrc[rix*rstride+t]=shared[t+rstride];
}


namespace cnine{


  template<typename CMAP>
  //template<typename CMAP, typename = typename std::enable_if<std::is_base_of<Direct_cmap,CMAP>::value, CMAP>::type>
  void CtensorA_plus_cu(const CMAP& cmap, CtensorArrayA& r, const CtensorArrayA& x, const CtensorArrayA& y, 
    const cudaStream_t& stream, const int add_flag){

      if(add_flag==0){
	CtensorA_plus_kernel<<<cmap.blockdims(),r.asize,0,stream>>>
	  (r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,r.cellstride,x.cellstride,y.cellstride,cmap);
      }
      if(add_flag==1){
	CtensorA_add_plus_kernel<<<cmap.blockdims(),r.asize,0,stream>>>
	  (r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,r.cellstride,x.cellstride,y.cellstride,cmap);
      }

  }


  template<typename CMAP>
  void CtensorA_plus_accumulator_cu(const CMAP& cmap, CtensorArrayA& r, const CtensorArrayA& x, const CtensorArrayA& y,
    const cudaStream_t& stream){

    CtensorA_plus_accumulator_kernel<<<cmap.blockdims(),r.asize,2*r.cellstride*sizeof(float),stream>>>
      (r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,r.cellstride,x.cellstride,y.cellstride,cmap);
    
  }

  template void CtensorA_plus_cu(const cnine::CellwiseBinaryCmap& map, 
    CtensorArrayA&, const CtensorArrayA&, const CtensorArrayA&, const cudaStream_t&, const int add_flag);

  template void CtensorA_plus_cu(const cnine::BroadcastBinaryCmap& map, 
    CtensorArrayA&, const CtensorArrayA&, const CtensorArrayA&, const cudaStream_t&, const int add_flag);

  template void CtensorA_plus_cu(const cnine::OuterCmap& map, 
    CtensorArrayA&, const CtensorArrayA&, const CtensorArrayA&, const cudaStream_t&, const int add_flag);



  template void CtensorA_plus_accumulator_cu(const cnine::InnerCmap& map, 
    CtensorArrayA&, const CtensorArrayA&, const CtensorArrayA&, const cudaStream_t&);

  template void CtensorA_plus_accumulator_cu(const cnine::MVprodCmap& map, 
    CtensorArrayA&, const CtensorArrayA&, const CtensorArrayA&, const cudaStream_t&);

  template void CtensorA_plus_accumulator_cu(const cnine::VMprodCmap& map, 
    CtensorArrayA&, const CtensorArrayA&, const CtensorArrayA&, const cudaStream_t&);

  template void CtensorA_plus_accumulator_cu(const cnine::Convolve2Cmap& map, 
    CtensorArrayA&, const CtensorArrayA&, const CtensorArrayA&, const cudaStream_t&);

  template void CtensorA_plus_accumulator_cu(const cnine::accumulate_cmap& map, 
    CtensorArrayA&, const CtensorArrayA&, const CtensorArrayA&, const cudaStream_t&);


}

#endif 


  //template void CtensorA_plus_cu(const cnine::accumulate_cmap& map, 
  //CtensorArrayA&, const CtensorArrayA&, const CtensorArrayA&, const cudaStream_t&, const int add_flag);

  /*
  template<typename CMAP>
  void CtensorA_add_plus_cu(const CMAP& cmap, CtensorArrayA& r, const CtensorArrayA& x, const CtensorArrayA& y,
    const cudaStream_t& stream){

    CtensorA_add_plus_kernel<<<cmap.blockdims(),r.asize,0,stream>>>
      (r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,r.cellstride,x.cellstride,y.cellstride,cmap);
    
  }
  */

  //template<typename CMAP, typename = typename std::enable_if<std::is_base_of<Masked2_cmap,CMAP>::value, CMAP>::type>
