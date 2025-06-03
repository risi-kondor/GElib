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

#ifndef _BasicCtensorProducts_cu
#define _BasicCtensorProducts_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

#include "Cnine_base.hpp"
#include "Ctensor2_view.hpp"


__global__ void BasicCproduct_4_1_3_kernel(const float* xarr, const float* xarrc, const float* yarr, const float* yarrc, float* rarr, float* rarrc, 
  const int xs0, const int xs1, const int xs2, const int xs3, 
  const int ys0, const int ys1, const int ys2, const int ys3, 
  const int rs0, const int rs1,  const int rs2, const int rs3){

  const int i0=blockIdx.x;
  const int i1=threadIdx.x;
  const int i2=threadIdx.y;
  const int i3=threadIdx.z;

  float xr=xarr[i0*xs0+i1*xs1+i2*xs2+i3*xs3];
  float xi=xarrc[i0*xs0+i1*xs1+i2*xs2+i3*xs3];
  float yr=yarr[i0*ys0+i1*ys1+i2*ys2+i3*ys3];
  float yi=yarrc[i0*ys0+i1*ys1+i2*ys2+i3*ys3];
  rarr[i0*rs0+i1*rs1+i2*rs2+i3*rs3]+=xr*yr-xi*yi;
  rarrc[i0*rs0+i1*rs1+i2*rs2+i3*rs3]+=xr*yi+xi*yr;
}


__global__ void BasicCproduct_4_2_2_kernel(const float* xarr, const float* xarrc, const float* yarr, const float* yarrc, float* rarr, float* rarrc, 
  const int xs0, const int xs1, const int xs2, const int xs3, 
  const int ys0, const int ys1, const int ys2, const int ys3, 
  const int rs0, const int rs1,  const int rs2, const int rs3){

  const int i0=blockIdx.x;
  const int i1=blockIdx.y;
  const int i2=threadIdx.x;
  const int i3=threadIdx.y;

  float xr=xarr[i0*xs0+i1*xs1+i2*xs2+i3*xs3];
  float xi=xarrc[i0*xs0+i1*xs1+i2*xs2+i3*xs3];
  float yr=yarr[i0*ys0+i1*ys1+i2*ys2+i3*ys3];
  float yi=yarrc[i0*ys0+i1*ys1+i2*ys2+i3*ys3];
  rarr[i0*rs0+i1*rs1+i2*rs2+i3*rs3]+=xr*yr-xi*yi;
  rarrc[i0*rs0+i1*rs1+i2*rs2+i3*rs3]+=xr*yi+xi*yr;
}


__global__ void BasicCproduct_4_3_1_kernel(const float* xarr, const float* xarrc, const float* yarr, const float* yarrc, float* rarr, float* rarrc, 
  const int xs0, const int xs1, const int xs2, const int xs3, 
  const int ys0, const int ys1, const int ys2, const int ys3, 
  const int rs0, const int rs1,  const int rs2, const int rs3){

  const int i0=blockIdx.x;
  const int i1=blockIdx.y;
  const int i2=blockIdx.z;
  const int i3=threadIdx.x;

  float xr=xarr[i0*xs0+i1*xs1+i2*xs2+i3*xs3];
  float xi=xarrc[i0*xs0+i1*xs1+i2*xs2+i3*xs3];
  float yr=yarr[i0*ys0+i1*ys1+i2*ys2+i3*ys3];
  float yi=yarrc[i0*ys0+i1*ys1+i2*ys2+i3*ys3];
  rarr[i0*rs0+i1*rs1+i2*rs2+i3*rs3]+=xr*yr-xi*yi;
  rarrc[i0*rs0+i1*rs1+i2*rs2+i3*rs3]+=xr*yi+xi*yr;
}


__global__ void BasicCproduct_2_1__0_2_kernel(const float* xarr, const float* xarrc, const float* yarr, const float* yarrc, float* rarr, float* rarrc, const int n2, 
  const int xs0, const int xs1, const int xs2,  
  const int ys0, const int ys1, const int ys2, 
  const int rs0, const int rs1){

  const int i0=threadIdx.x;
  const int i1=threadIdx.y;

  float rr=0;
  float ri=0;

  for(int i2=0; i2<n2; i2++){
    float xr=xarr[i0*xs0+i1*xs1+i2*xs2];
    float xi=xarrc[i0*xs0+i1*xs1+i2*xs2];
    float yr=yarr[i0*ys0+i1*ys1+i2*ys2];
    float yi=yarrc[i0*ys0+i1*ys1+i2*ys2];
    rr+=xr*yr-xi*yi;
    ri+=xr*yi+xi*yr;
  }
  rarr[i0*rs0+i1*rs1]+=rr;
  rarrc[i0*rs0+i1*rs1]+=ri;
}


__global__ void BasicCproduct_2_1__1_1_kernel(const float* xarr, const float* xarrc, const float* yarr, const float* yarrc, float* rarr, float* rarrc, const int n2, 
  const int xs0, const int xs1, const int xs2,  
  const int ys0, const int ys1, const int ys2, 
  const int rs0, const int rs1){

  const int i0=blockIdx.x;
  const int i1=threadIdx.x;

  float rr=0;
  float ri=0;

  for(int i2=0; i2<n2; i2++){
    float xr=xarr[i0*xs0+i1*xs1+i2*xs2];
    float xi=xarrc[i0*xs0+i1*xs1+i2*xs2];
    float yr=yarr[i0*ys0+i1*ys1+i2*ys2];
    float yi=yarrc[i0*ys0+i1*ys1+i2*ys2];
    rr+=xr*yr-xi*yi;
    ri+=xr*yi+xi*yr;
  }
  rarr[i0*rs0+i1*rs1]+=rr;
  rarrc[i0*rs0+i1*rs1]+=ri;
}


__global__ void Ctensor2_add_otimes_kernel(const cnine::Ctensor2_view r, const cnine::Ctensor2_view x, const cnine::Ctensor2_view y, const float c){
  const int t=blockIdx.x*blockDim.x+threadIdx.x;
  const int i0=t/r.n1;
  const int i1=t%r.n1;
  if(i0>=r.n0) return;
  float xr=x.arr[i0*x.s0+i1*x.s1];
  float xi=x.arrc[i0*x.s0+i1*x.s1];
  float yr=y.arr[i0*y.s0+i1*y.s1];
  float yi=y.arrc[i0*y.s0+i1*y.s1];
  r.arr[i0*r.s0+i1*r.s1]+=(xr*yr-xi*yi)*c;
  r.arrc[i0*r.s0+i1*r.s1]+=(xr*yi+xi*yr)*c;
}

__global__ void Ctensor2_add_otimesc_kernel(const cnine::Ctensor2_view r, const cnine::Ctensor2_view x, const cnine::Ctensor2_view y, const float c){
  const int t=blockIdx.x*blockDim.x+threadIdx.x;
  const int i0=t/r.n1;
  const int i1=t%r.n1;
  if(i0>=r.n0) return;
  float xr=x.arr[i0*x.s0+i1*x.s1];
  float xi=x.arrc[i0*x.s0+i1*x.s1];
  float yr=y.arr[i0*y.s0+i1*y.s1];
  float yi=y.arrc[i0*y.s0+i1*y.s1];
  r.arr[i0*r.s0+i1*r.s1]+=(xr*yr+xi*yi)*c;
  r.arrc[i0*r.s0+i1*r.s1]+=(-xr*yi+xi*yr)*c;
}


namespace cnine{

  void BasicCproduct_4_cu(const float* xarr, const float* xarrc, const float* yarr, const float* yarrc, float* rarr, float* rarrc,
    const int n0, const int n1, const int n2, const int n3, 
    const int xs0, const int xs1, const int xs2, const int xs3, 
    const int ys0, const int ys1, const int ys2, const int ys3, 
    const int rs0, const int rs1,  const int rs2, const int rs3, 
    const cudaStream_t& stream){

    if(n1*n2*n3<=1024){
      dim3 blocks(n0);
      dim3 threads(n1,n2,n3);
      BasicCproduct_4_1_3_kernel<<<blocks,threads,0,stream>>>(xarr,xarrc,yarr,yarrc,rarr,rarrc,
	xs0,xs1,xs2,xs3,ys0,ys1,ys2,ys3,rs0,rs1,rs2,rs3);
      return;
    }

    if(n2*n3<=1024){
      dim3 blocks(n0,n1);
      dim3 threads(n2,n3);
      BasicCproduct_4_2_2_kernel<<<blocks,threads,0,stream>>>(xarr,xarrc,yarr,yarrc,rarr,rarrc,
	xs0,xs1,xs2,xs3,ys0,ys1,ys2,ys3,rs0,rs1,rs2,rs3);
      return;
    }

    if(n3<=1024){
      dim3 blocks(n0,n1,n2);
      dim3 threads(n3);
      BasicCproduct_4_3_1_kernel<<<blocks,threads,0,stream>>>(xarr,xarrc,yarr,yarrc,rarr,rarrc,
	xs0,xs1,xs2,xs3,ys0,ys1,ys2,ys3,rs0,rs1,rs2,rs3);
      return;
    }

    CNINE_COUT("Error: tensor too large for BasicCproduct_4 kernel");

  }


  void BasicCproduct_2_1_cu(const float* xarr, const float* xarrc, const float* yarr, const float* yarrc, float* rarr, float* rarrc, 
    const int n0, const int n1, const int n2,  
    const int xs0, const int xs1, const int xs2, 
    const int ys0, const int ys1, const int ys2, 
    const int rs0, const int rs1,  
    const cudaStream_t& stream){

    if(n0*n1<=1024){
      dim3 threads(n0,n1);
      BasicCproduct_2_1__0_2_kernel<<<1,threads,0,stream>>>(xarr,xarrc,yarr,yarrc,rarr,rarrc,n2,
	xs0,xs1,xs2,ys0,ys1,ys2,rs0,rs1);
      return;
    }

    if(n1<=1024){
      dim3 blocks(n0);
      dim3 threads(n1);
      BasicCproduct_2_1__1_1_kernel<<<blocks,threads,0,stream>>>(xarr,xarrc,yarr,yarrc,rarr,rarrc,n2,
	xs0,xs1,xs2,ys0,ys1,ys2,rs0,rs1);
      return;
    }

    CNINE_COUT("Error: tensor too large for BasicCproduct_2_1 kernel");

  }


  void Ctensor2_add_otimes_cu(const Ctensor2_view& r, const Ctensor2_view& x, const Ctensor2_view& y, const float c, 
    const cudaStream_t& stream){
    CNINE_ASSRT(r.n0==x.n0);
    CNINE_ASSRT(r.n0==y.n0);
    CNINE_ASSRT(r.n1==x.n1);
    CNINE_ASSRT(r.n1==y.n1);
    Ctensor2_add_otimes_kernel<<<roundup(r.n0*r.n1,32)/32,32,0,stream>>>(r,x,y,c);
  }

  void Ctensor2_add_otimesc_cu(const Ctensor2_view& r, const Ctensor2_view& x, const Ctensor2_view& y, const float c, 
    const cudaStream_t& stream){
    CNINE_ASSRT(r.n0==x.n0);
    CNINE_ASSRT(r.n0==y.n0);
    CNINE_ASSRT(r.n1==x.n1);
    CNINE_ASSRT(r.n1==y.n1);
    Ctensor2_add_otimesc_kernel<<<roundup(r.n0*r.n1,32)/32,32,0,stream>>>(r,x,y,c);
  }


}

#endif 
