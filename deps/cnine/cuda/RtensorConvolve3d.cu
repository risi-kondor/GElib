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

#ifndef _RtensorConvolve3d_cu
#define _RtensorConvolve3d_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

#include "Cnine_base.hpp"
#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"
#include "Rtensor6_view.hpp"
#include "Itensor1_view.hpp"
#include "Itensor2_view.hpp"
#include "CUDAhelpers.hpp"


// ---- 4D case: (i0,i1,i2,a)*(a',j0,j1,j2,a) -> (i0+j0,i1+j1,i2+j2,a') --------------------------------------


__global__ void RtensorConvolve3d_kernel
(float* rarr, const int rs0, const int rs1, const int rs2, const int rs3,   
  float* xarr, const int xs0, const int xs1, const int xs2, const int xs3,
  float* warr, const int ws0, const int ws1, const int ws2, const int ws3, const int ws4, 
  const int nj0, const int nj1, const int nj2, const int na){

  int i0=blockIdx.x;
  int i1=blockIdx.y;
  int i2=blockIdx.z;

  float t=0;
  for(int j0=0; j0<nj0; j0++)
    for(int j1=0; j1<nj1; j1++)
      for(int j2=0; j2<nj2; j2++)
	for(int a=0; a<na; a++)
	  t+=xarr[(i0+j0)*xs0+(i1+j1)*xs1+(i2+j2)*xs2+a*xs3]*
	    (*(warr+threadIdx.x*ws0+j0*ws1+j1*ws2+j2*ws3+a*ws4));

  rarr[i0*rs0+i1*rs1+i2*rs2+threadIdx.x*rs3]+=t;
}

__global__ void RtensorConvolve3d_kernel
(float* rarr, const int rs0, const int rs1, const int rs2, const int rs3,   
  float* xarr, const int xs0, const int xs1, const int xs2, const int xs3,
  float* warr, const int ws0, const int ws1, const int ws2, const int ws3, const int ws4, 
  const int nj0, const int nj1, const int nj2, const int na,
  const int xn0, const int xn1, const int xn2, const int padding0, const int padding1, const int padding2){

  int i0=blockIdx.x;
  int i1=blockIdx.y;
  int i2=blockIdx.z;

  float t=0;
  for(int j0=max(0,padding0-i0); j0<min(nj0,xn0-i0+padding0); j0++)
    for(int j1=max(0,padding1-i1); j1<min(nj1,xn1-i1+padding1); j1++)
      for(int j2=max(0,padding2-i2); j2<min(nj2,xn2-i2+padding2); j2++)
	for(int a=0; a<na; a++)
	  t+=xarr[(i0+j0-padding0)*xs0+(i1+j1-padding1)*xs1+(i2+j2-padding2)*xs2+a*xs3]*
	    (*(warr+threadIdx.x*ws0+j0*ws1+j1*ws2+j2*ws3+a*ws4));

  rarr[i0*rs0+i1*rs1+i2*rs2+threadIdx.x*rs3]+=t;
}


// ---- 5D case (i0,i1,i2,a,c)*(a',j0,j1,j2,a) -> (i0+j0,i1+j1,i2+j2,a',c) -----------------------------------


__global__ void RtensorConvolve3d_kernel
(float* rarr, const int rs0, const int rs1, const int rs2, const int rs3, const int rs4,   
  float* xarr, const int xs0, const int xs1, const int xs2, const int xs3, const int xs4, 
  float* warr, const int ws0, const int ws1, const int ws2, const int ws3, const int ws4, 
  const int nj0, const int nj1, const int nj2, const int na){

  int i0=blockIdx.x;
  int i1=blockIdx.y;
  int i2=blockIdx.z;

  float t=0;
  for(int j0=0; j0<nj0; j0++)
    for(int j1=0; j1<nj1; j1++)
      for(int j2=0; j2<nj2; j2++)
	for(int a=0; a<na; a++)
	  t+=xarr[(i0+j0)*xs0+(i1+j1)*xs1+(i2+j2)*xs2+a*xs3+threadIdx.y*xs4]*
	    (*(warr+threadIdx.x*ws0+j0*ws1+j1*ws2+j2*ws3+a*ws4));

  rarr[i0*rs0+i1*rs1+i2*rs2+threadIdx.x*rs3+threadIdx.y*rs4]+=t;
}


__global__ void RtensorConvolve3d_kernel
(float* rarr, const int rs0, const int rs1, const int rs2, const int rs3, const int rs4,   
  float* xarr, const int xs0, const int xs1, const int xs2, const int xs3, const int xs4, 
  float* warr, const int ws0, const int ws1, const int ws2, const int ws3, const int ws4, 
  const int nj0, const int nj1, const int nj2, const int na,
  const int xn0, const int xn1, const int xn2, const int padding0, const int padding1, const int padding2){

  int i0=blockIdx.x;
  int i1=blockIdx.y;
  int i2=blockIdx.z;

  float t=0;
  for(int j0=max(0,padding0-i0); j0<min(nj0,xn0-i0+padding0); j0++)
    for(int j1=max(0,padding1-i1); j1<min(nj1,xn1-i1+padding1); j1++)
      for(int j2=max(0,padding2-i2); j2<min(nj2,xn2-i2+padding2); j2++)
	for(int a=0; a<na; a++)
	  t+=xarr[(i0+j0-padding0)*xs0+(i1+j1-padding1)*xs1+(i2+j2-padding2)*xs2+a*xs3+threadIdx.y*xs4]*
	    (*(warr+threadIdx.x*ws0+j0*ws1+j1*ws2+j2*ws3+a*ws4));

  rarr[i0*rs0+i1*rs1+i2*rs2+threadIdx.x*rs3+threadIdx.y*rs4]+=t;
}


// ---- 6D case (b,i0,i1,i2,a,c)*(a',j0,j1,j2,a) -> (b,i0+j0,i1+j1,i2+j2,a',c) --------------------------------


__global__ void RtensorConvolve3d_kernel
(float* rarr, const int rs0, const int rs1, const int rs2, const int rs3, const int rs4, const int rs5,   
  float* xarr, const int xs0, const int xs1, const int xs2, const int xs3, const int xs4, const int xs5, 
  float* warr, const int ws0, const int ws1, const int ws2, const int ws3, const int ws4, 
  const int rn1, const int nj0, const int nj1, const int nj2, const int na){

  int b=blockIdx.x/rn1;
  int i0=blockIdx.x%rn1;
  int i1=blockIdx.y;
  int i2=blockIdx.z;

  float t=0;
  for(int j0=0; j0<nj0; j0++)
    for(int j1=0; j1<nj1; j1++)
      for(int j2=0; j2<nj2; j2++)
	for(int a=0; a<na; a++)
	  t+=xarr[b*xs0+(i0+j0)*xs1+(i1+j1)*xs2+(i2+j2)*xs3+a*xs4+threadIdx.y*xs5]*
	    (*(warr+threadIdx.x*ws0+j0*ws1+j1*ws2+j2*ws3+a*ws4));

  rarr[b*rs0+i0*rs1+i1*rs2+i2*rs3+threadIdx.x*rs4+threadIdx.y*rs5]+=t;
}


__global__ void RtensorConvolve3d_kernel
(float* rarr, const int rs0, const int rs1, const int rs2, const int rs3, const int rs4, const int rs5,   
  float* xarr, const int xs0, const int xs1, const int xs2, const int xs3, const int xs4, const int xs5, 
  float* warr, const int ws0, const int ws1, const int ws2, const int ws3, const int ws4, 
  const int rn1, const int nj0, const int nj1, const int nj2, const int na,
  const int xn0, const int xn1, const int xn2, const int padding0, const int padding1, const int padding2){

  int b=blockIdx.x/rn1;
  int i0=blockIdx.x%rn1;
  int i1=blockIdx.y;
  int i2=blockIdx.z;

  float t=0;
  for(int j0=max(0,padding0-i0); j0<min(nj0,xn0-i0+padding0); j0++)
    for(int j1=max(0,padding1-i1); j1<min(nj1,xn1-i1+padding1); j1++)
      for(int j2=max(0,padding2-i2); j2<min(nj2,xn2-i2+padding2); j2++)
	for(int a=0; a<na; a++)
	  t+=xarr[b*xs0+(i0+j0-padding0)*xs1+(i1+j1-padding1)*xs2+(i2+j2-padding2)*xs3+a*xs4+threadIdx.y*xs5]*
	    (*(warr+threadIdx.x*ws0+j0*ws1+j1*ws2+j2*ws3+a*ws4)); 
  // CUDA apparently doesn't support negative array indices

  rarr[b*rs0+i0*rs1+i1*rs2+i2*rs3+threadIdx.x*rs4+threadIdx.y*rs5]+=t;
}


// ----------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------


namespace cnine{


  // ---- 4D case: (i0,i1,i2,a)*(a',j0,j1,j2,a) -> (i0+j0,i1+j1,i2+j2,a') --------------------------------------


  void RtensorConvolve3d_cu(const Rtensor4_view& r, const Rtensor4_view& x, const Rtensor5_view& w, 
    const int padding0, const int padding1, const int padding2, const cudaStream_t& stream){
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(w.dev==1);

    dim3 blocks(r.n0,r.n1,r.n2);
    dim3 threads(r.n3);

    if(padding0==0&&padding1==0&&padding2==0){
      RtensorConvolve3d_kernel<<<blocks,threads,0,stream>>>
	(r.arr,r.s0,r.s1,r.s2,r.s3,
	  x.arr,x.s0,x.s1,x.s2,x.s3,
	  w.arr,w.s0,w.s1,w.s2,w.s3,w.s4,
	  w.n1,w.n2,w.n3,w.n4); 
    }else{
      RtensorConvolve3d_kernel<<<blocks,threads,0,stream>>>
	(r.arr,r.s0,r.s1,r.s2,r.s3,
	  x.arr,x.s0,x.s1,x.s2,x.s3,
	  w.arr,w.s0,w.s1,w.s2,w.s3,w.s4,
	  w.n1,w.n2,w.n3,w.n4,
	  x.n0,x.n1,x.n2,padding0,padding1,padding2); 
    }
  }
    

  // ---- 5D case (i0,i1,i2,a,c)*(a',j0,j1,j2,a) -> (i0+j0,i1+j1,i2+j2,a',c) -----------------------------------


  void RtensorConvolve3d_cu(const Rtensor5_view& r, const Rtensor5_view& x, const Rtensor5_view& w, 
    const int padding0, const int padding1, const int padding2, const cudaStream_t& stream){
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(w.dev==1);

    if(r.n3*r.n4<=1024){
      dim3 blocks(r.n0,r.n1,r.n2);
      dim3 threads(r.n3,r.n4);

      if(padding0==0&&padding1==0&&padding2==0){
	RtensorConvolve3d_kernel<<<blocks,threads,0,stream>>>
	  (r.arr,r.s0,r.s1,r.s2,r.s3,r.s4,
	    x.arr,x.s0,x.s1,x.s2,x.s3,x.s4,
	    w.arr,w.s0,w.s1,w.s2,w.s3,w.s4,
	    w.n1,w.n2,w.n3,w.n4); 
      }else{
	RtensorConvolve3d_kernel<<<blocks,threads,0,stream>>>
	  (r.arr,r.s0,r.s1,r.s2,r.s3,r.s4,
	    x.arr,x.s0,x.s1,x.s2,x.s3,x.s4,
	    w.arr,w.s0,w.s1,w.s2,w.s3,w.s4,
	    w.n1,w.n2,w.n3,w.n4,
	    x.n0,x.n1,x.n2,padding0,padding1,padding2); 
      }

    }else{
      cout<<"currently unimplemented"<<endl;
    }

  }


  // ---- 6D case (b,i0,i1,i2,a,c)*(a',j0,j1,j2,a) -> (b,i0+j0,i1+j1,i2+j2,a',c) -----------------------------


  void RtensorConvolve3d_cu(const Rtensor6_view& r, const Rtensor6_view& x, const Rtensor5_view& w, 
    const int padding0, const int padding1, const int padding2, const cudaStream_t& stream){
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(w.dev==1);

    if(r.n4*r.n5<=1024){
      dim3 blocks(r.n0*r.n1,r.n2,r.n3);
      dim3 threads(r.n4,r.n5);

      if(padding0==0&&padding1==0&&padding2==0){
	RtensorConvolve3d_kernel<<<blocks,threads,0,stream>>>
	  (r.arr,r.s0,r.s1,r.s2,r.s3,r.s4,r.s5,
	    x.arr,x.s0,x.s1,x.s2,x.s3,x.s4,x.s5,
	    w.arr,w.s0,w.s1,w.s2,w.s3,w.s4,
	    r.n1,w.n1,w.n2,w.n3,w.n4); 
      }else{
	RtensorConvolve3d_kernel<<<blocks,threads,0,stream>>>
	  (r.arr,r.s0,r.s1,r.s2,r.s3,r.s4,r.s5,
	    x.arr,x.s0,x.s1,x.s2,x.s3,x.s4,x.s5,
	    w.arr,w.s0,w.s1,w.s2,w.s3,w.s4,
	    r.n1,w.n1,w.n2,w.n3,w.n4,
	    x.n1,x.n2,x.n3,padding0,padding1,padding2); 
      }

    }else{
      cout<<"currently unimplemented"<<endl;
    }

  }

}

#endif 
