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


#ifndef _Einsum2_cu
#define _Einsum2_cu

#include <cuda.h>
#include <cuda_runtime.h>

#include "TensorView.hpp"
#include "EinsumParams.hpp"


template<typename TYPE>
__global__ void add_einsum2_cu(TYPE* rarr, TYPE* xarr, TYPE* yarr, Einsum2params p){

  TYPE* rarr_t=rarr+blockIdx.x*p.tstride_x[0]+blockIdx.y*p.tstride_x[1]+blockIdx.z*p.tstride_x[2];
  TYPE* xarr_t=xarr+blockIdx.x*p.tstride_y[0]+blockIdx.y*p.tstride_y[1]+blockIdx.z*p.tstride_y[2];
  TYPE* yarr_t=yarr+blockIdx.x*p.tstride_r[0]+blockIdx.y*p.tstride_r[1]+blockIdx.z*p.tstride_r[2];
  
  TYPE t=0;
  // contraction loops
  for(int c0=0; c0<p.cdims[0]; c0++)
    for(int c1=0; c1<p.cdims[1]; c1++)
      for(int c2=0; c2<p.cdims[2]; c2++){
	TYPE* xarr_c=xarr_t+c0*p.cstride_x[0]+c1*p.cstride_x[1]+c2*p.cstride_x[2];
	TYPE* yarr_c=yarr_t+c0*p.cstride_y[0]+c1*p.cstride_y[1]+c2*p.cstride_y[2];

	TYPE xt=0;
	for(int xs0=0; xs0<p.xsdims[0]; xs0++)
	  for(int xs1=0; xs1<p.xsdims[1]; xs1++)
	    for(int xs2=0; xs2<p.xsdims[2]; xs2++)
	      xt+=*(xarr_c+xs0*p.xsstride[0]+xs1*p.xsstride[1]+xs2*p.xsstride[2]);
	
	TYPE yt=0;
	for(int ys0=0; ys0<p.ysdims[0]; ys0++)
	  for(int ys1=0; ys1<p.ysdims[1]; ys1++)
	    for(int ys2=0; ys2<p.ysdims[2]; ys2++)
	      yt+=*(yarr_c+ys0*p.ysstride[0]+ys1*p.ysstride[1]+ys2*p.ysstride[2]);
	
	t+=xt*yt;
      }
	      
  // broadcast loops
  for(int b0=0; b0<p.bdims[0]; b0++)
    for(int b1=0; b1<p.bdims[1]; b1++)
      for(int b2=0; b2<p.bdims[2]; b2++)
	*(rarr_t+b0*p.bstride[0]+b1*p.bstride[1]+b2*p.bstride[2])+=t;
}


namespace cnine{

  template<typename TYPE>
  void add_einsum2_cu(const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y, 
    const Einsum2params& params, const cudaStream_t& stream){
    CNINE_ASSRT(r.get_dims()==x.get_dims());
    int D=r.ndims();

    CNINE_ASSRT(params.tdims[3]==1); // for now
    dim3 blocks(params.tdims[0],params.tdims[1],params.tdims[2]);
    add_einsum2_kernel<<<blocks,1,0,stream>>>(r.get_arr(),x.get_arr(),y.get_arr(),params);

  }

  template<>
  void add_einsum2_cu(const TensorView<float>& r, const TensorView<float>& x, const TensorView<float>& y, 
    const Einsum2params& params, const cudaStream_t& stream);

}
#endif 
