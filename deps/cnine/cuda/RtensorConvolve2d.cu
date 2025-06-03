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

#ifndef _RtensorConvolve2d_cu
#define _RtensorConvolve2d_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

#include "Cnine_base.hpp"
#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"
#include "Rtensor5_view.hpp"
#include "Itensor1_view.hpp"
#include "Itensor2_view.hpp"
#include "CUDAhelpers.hpp"


// 4D case 
__global__ void RtensorConvolve2d_kernel(float* rarr, const int rs0, const int rs1, const int rs2, const int rs3,  
  float* xarr, const int xs0, const int xs1, const int xs2, const int xs3, 
  float* warr, const int ws0, const int ws1, const int ws2, const int ws3, 
  const int nj0, const int nj1, const int na){

  int i0=blockIdx.x;
  int i1=blockIdx.y;

  float t=0;
  for(int j0=0; j0<nj0; j0++)
    for(int j1=0; j1<nj1; j1++)
      for(int a=0; a<na; a++)
	t+=xarr[(i0+j0)*xs0+(i1+j1)*xs1+a*xs2+threadIdx.x*xs3]*
	  (*(warr+blockIdx.z*ws0+j0*ws1+j1*ws2+a*ws3));

  rarr[i0*rs0+i1*rs1+blockIdx.z*rs2+threadIdx.x*rs3]+=t;
}


__global__ void RtensorConvolve2d_kernel(float* rarr, const int rs0, const int rs1, const int rs2, const int rs3,  
  float* xarr, const int xs0, const int xs1, const int xs2, const int xs3, 
  float* warr, const int ws0, const int ws1, const int ws2, const int ws3, 
  const int nj0, const int nj1, const int na,
  const int xn0, const int xn1, const int padding0, const int padding1){
  
  int i0=blockIdx.x;
  int i1=blockIdx.y;

  float t=0;
  for(int j0=max(0,padding0-i0); j0<min(nj0,xn0-i0+padding0); j0++)
    for(int j1=max(0,padding1-i1); j1<min(nj1,xn1-i1+padding1); j1++)
      for(int a=0; a<na; a++)
	t+=xarr[(i0+j0-padding0)*xs0+(i1+j1-padding1)*xs1+a*xs2+threadIdx.x*xs3]*
	  (*(warr+blockIdx.z*ws0+j0*ws1+j1*ws2+a*ws3));

  rarr[i0*rs0+i1*rs1+blockIdx.z*rs2+threadIdx.x*rs3]+=t;
}


// 5D case
__global__ void RtensorConvolve2d_kernel(float* rarr, const int rs0, const int rs1, const int rs2, const int rs3, const int rs4, 
  float* xarr, const int xs0, const int xs1, const int xs2, const int xs3, const int xs4,  
  float* warr, const int ws0, const int ws1, const int ws2, const int ws3, 
  const int rn2, const int nj0, const int nj1, const int na){

  int i0=blockIdx.y/rn2;
  int i1=blockIdx.y%rn2;

  float t=0;
  for(int j0=0; j0<nj0; j0++)
    for(int j1=0; j1<nj1; j1++)
      for(int a=0; a<na; a++)
	t+=xarr[blockIdx.x*xs0+(i0+j0)*xs1+(i1+j1)*xs2+a*xs3+threadIdx.x*xs4]*
	  (*(warr+blockIdx.z*ws0+j0*ws1+j1*ws2+a*ws3));

  rarr[blockIdx.x*rs0+i0*rs1+i1*rs2+blockIdx.z*rs3+threadIdx.x*rs4]+=t;
}


__global__ void RtensorConvolve2d_kernel(float* rarr, const int rs0, const int rs1, const int rs2, const int rs3, const int rs4, 
  float* xarr, const int xs0, const int xs1, const int xs2, const int xs3, const int xs4,  
  float* warr, const int ws0, const int ws1, const int ws2, const int ws3, 
  const int rn2, const int nj0, const int nj1, const int na, 
  const int xn0, const int xn1, const int padding0, const int padding1){

  int i0=blockIdx.y/rn2;
  int i1=blockIdx.y%rn2;

  float t=0;
  for(int j0=max(0,padding0-i0); j0<min(nj0,xn0-i0+padding0); j0++)
    for(int j1=max(0,padding1-i1); j1<min(nj1,xn1-i1+padding1); j1++)
      for(int a=0; a<na; a++)
	t+=xarr[blockIdx.x*xs0+(i0+j0-padding0)*xs1+(i1+j1-padding1)*xs2+a*xs3+threadIdx.x*xs4]*
	  (*(warr+blockIdx.z*ws0+j0*ws1+j1*ws2+a*ws3));

  rarr[blockIdx.x*rs0+i0*rs1+i1*rs2+blockIdx.z*rs3+threadIdx.x*rs4]+=t;
}


// ----------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------


namespace cnine{


  void RtensorConvolve2d_cu(const Rtensor4_view& r, const Rtensor4_view& x, const Rtensor4_view& w, 
    const int padding0, const int padding1, const cudaStream_t& stream){
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(w.dev==1);

    dim3 blocks(r.n0,r.n1,r.n2);

    if(padding0==0&&padding1==0){
      RtensorConvolve2d_kernel<<<blocks,r.n3,0,stream>>>
	(r.arr,r.s0,r.s1,r.s2,r.s3,
	  x.arr,x.s0,x.s1,x.s2,x.s3,
	  w.arr,w.s0,w.s1,w.s2,w.s3,
	  w.n1,w.n2,w.n3);
    }else{
     RtensorConvolve2d_kernel<<<blocks,r.n3,0,stream>>>
	(r.arr,r.s0,r.s1,r.s2,r.s3,
	  x.arr,x.s0,x.s1,x.s2,x.s3,
	  w.arr,w.s0,w.s1,w.s2,w.s3,
	  w.n1,w.n2,w.n3,
	  x.n0,x.n1,padding0,padding1); // changed from x.n1, x.n2
    }
  }

  void RtensorConvolve2d_cu(const Rtensor5_view& r, const Rtensor5_view& x, const Rtensor4_view& w, 
    const int padding0, const int padding1, const cudaStream_t& stream){
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(w.dev==1);

    dim3 blocks(r.n0,r.n1*r.n2,r.n3);

    if(padding0==0&&padding1==0){
      RtensorConvolve2d_kernel<<<blocks,r.n4,0,stream>>>
	(r.arr,r.s0,r.s1,r.s2,r.s3,r.s4,
	  x.arr,x.s0,x.s1,x.s2,x.s3,x.s4,
	  w.arr,w.s0,w.s1,w.s2,w.s3,
	  r.n2,w.n1,w.n2,w.n3); // changed to r.n2??
    }else{
     RtensorConvolve2d_kernel<<<blocks,r.n4,0,stream>>>
	(r.arr,r.s0,r.s1,r.s2,r.s3,r.s4,
	  x.arr,x.s0,x.s1,x.s2,x.s3,x.s4,
	  w.arr,w.s0,w.s1,w.s2,w.s3,
	  r.n2,w.n1,w.n2,w.n3, // changed to r.n2
	  x.n1,x.n2,padding0,padding1); 
    }
  }

}

#endif 
