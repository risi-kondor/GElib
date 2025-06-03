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

#ifndef _CtensorA_add_plus_cu
#define _CtensorA_add_plus_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

#include "Cmaps.hpp"
#include "CtensorArrayA.hpp"


template<typename IMAP>
__global__ void CtensorA_add_plus_kernel(float* rarr, float* rarrc, float* xarr, float* xarrc, 
  float* yarr, float* yarrc, const int rstride, const int xstride, const int ystride, const IMAP& map){
  
  auto T=map(blockIdx.x,blockIdx.y,blockIdx.z);
  const int rix=thrust::get<0>(T);
  const int xix=thrust::get<1>(T);
  const int yix=thrust::get<2>(T);
  const int t=threadIdx.x;

  rarr[rix*rstride+t]+=xarr[xix*xstride+t]+yarr[yix*ystride+t];
  rarrc[rix*rstride+t]+=xarrc[xix*xstride+t]+yarr[yix*ystride+t];
}


namespace cnine{

  template<typename CMAP>
  void CtensorA_add_plus_cu(const CMAP& map, CtensorArrayA& r, const CtensorArrayA& x, const CtensorArrayA& y, 
    const cudaStream_t& stream){

    CtensorA_add_plus_kernel<<<map.blockdims(),r.asize,0,stream>>>
      (r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,r.cellstride,x.cellstride,y.cellstride,map);
    
  }

  template void CtensorA_add_plus_cu<cnine::CellwiseBiCmap>(const cnine::CellwiseBiCmap& map, 
    CtensorArrayA&, const CtensorArrayA&, const CtensorArrayA&, const cudaStream_t&);

  template void CtensorA_add_plus_cu(const cnine::BroadcastLeftBiCmap& map, 
    CtensorArrayA&, const CtensorArrayA&, const CtensorArrayA&, const cudaStream_t&);

  template void CtensorA_add_plus_cu(const cnine::BroadcastRightBiCmap& map, 
    CtensorArrayA&, const CtensorArrayA&, const CtensorArrayA&, const cudaStream_t&);

  template void CtensorA_add_plus_cu(const cnine::InnerBiCmap& map, 
    CtensorArrayA&, const CtensorArrayA&, const CtensorArrayA&, const cudaStream_t&);

  template void CtensorA_add_plus_cu(const cnine::OuterBiCmap& map, 
    CtensorArrayA&, const CtensorArrayA&, const CtensorArrayA&, const cudaStream_t&);

  template void CtensorA_add_plus_cu(const cnine::MprodBiCmap& map, 
    CtensorArrayA&, const CtensorArrayA&, const CtensorArrayA&, const cudaStream_t&);

  template void CtensorA_add_plus_cu(const cnine::Convolve1BiCmap& map, 
    CtensorArrayA&, const CtensorArrayA&, const CtensorArrayA&, const cudaStream_t&);

  template void CtensorA_add_plus_cu(const cnine::Convolve2BiCmap& map, 
    CtensorArrayA&, const CtensorArrayA&, const CtensorArrayA&, const cudaStream_t&);

}

#endif 
