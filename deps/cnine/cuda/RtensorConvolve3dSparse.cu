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

#ifndef _RtensorConvolveSparse_cu
#define _RtensorConvolveSparse_cu

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
#include "CSRmatrix.hpp"
#include "CUDAhelpers.hpp"


// ---- 4D case (i0,i1,i2,a)*(a',j0,j1,j2,a) -> (i0+j0,i1+j1,i2+j2,a') ---------------------------------------

__global__ void RtensorConvolve3d_sparse_kernel
(const cnine::Rtensor4_view r, const cnine::Rtensor4_view x, float* warr, int* wdir, 
  const int J0, const int J1, const int J2){

  int i0=blockIdx.x/(r.n1*r.n2);
  int i1=(blockIdx.x/r.n2)-i0*r.n1;
  int i2=blockIdx.x%r.n2;

  int offs=wdir[2*threadIdx.x];
  int n=wdir[2*threadIdx.x+1];
  int na=x.n3;

  float t=0;
  for(int i=0; i<n; i++){
    int s=*reinterpret_cast<int*>(warr+offs+2*i);
    int j0=s/(J1*J2*na);
    int j1=s/(J2*na)-j0*J1;
    int j2=s/na-(j0*J1*J2+j1*J2);
    int a=s%na;
    t+=x.arr[(i0+j0)*x.s0+(i1+j1)*x.s1+(i2+j2)*x.s2+a*x.s3]*warr[offs+2*i+1];
  }
  r.arr[i0*r.s0+i1*r.s1+i2*r.s2+blockIdx.y*r.s3]+=t;
  
}


__global__ void RtensorConvolve3d_sparse_kernel
(const cnine::Rtensor4_view r, const cnine::Rtensor4_view x, float* warr, int* wdir, 
  const int J0, const int J1, const int J2, const int padding0, const int padding1, const int padding2){

  int i0=blockIdx.x/(r.n1*r.n2);
  int i1=(blockIdx.x/r.n2)-i0*r.n1;
  int i2=blockIdx.x%r.n2;

  int offs=wdir[2*threadIdx.x];
  int n=wdir[2*threadIdx.x+1];
  int na=x.n3;

  float t=0;
  for(int i=0; i<n; i++){
    int s=*reinterpret_cast<int*>(warr+offs+2*i);
    int j0=s/(J1*J2*na);
    int j1=s/(J2*na)-j0*J1;
    int j2=s/na-(j0*J1*J2+j1*J2);
    int a=s%na;
    if(i0+j0-padding0<0 || i0+j0-padding0>=x.n0) continue;
    if(i1+j1-padding1<0 || i1+j1-padding1>=x.n1) continue;
    if(i2+j2-padding1<0 || i2+j2-padding2>=x.n2) continue;
    t+=x.arr[(i0+j0)*x.s0+(i1+j1)*x.s1+(i2+j2)*x.s2+a*x.s3]*warr[offs+2*i+1];
  }
  r.arr[i0*r.s0+i1*r.s1+i2*r.s2+blockIdx.y*r.s3]+=t;
  
}


// ---- 5D case (i0,i1,i2,a,c)*(a',j0,j1,j2,a) -> (i0+j0,i1+j1,i2+j2,a',c) -----------------------------------

__global__ void RtensorConvolve3d_sparse_kernel
(const cnine::Rtensor5_view r, const cnine::Rtensor5_view x, float* warr, int* wdir, 
  const int J0, const int J1, const int J2){

  int i0=blockIdx.x/(r.n1*r.n2);
  int i1=(blockIdx.x/r.n2)-i0*r.n1;
  int i2=blockIdx.x%r.n2;

  int offs=wdir[2*blockIdx.y];
  int n=wdir[2*blockIdx.y+1];
  int na=x.n3;

  float t=0;
  for(int i=0; i<n; i++){
    int s=*reinterpret_cast<int*>(warr+offs+2*i);
    int j0=s/(J1*J2*na);
    int j1=s/(J2*na)-j0*J1;
    int j2=s/na-(j0*J1*J2+j1*J2);
    int a=s%na;
    t+=x.arr[(i0+j0)*x.s0+(i1+j1)*x.s1+(i2+j2)*x.s2+a*x.s3+threadIdx.x*x.s4]*warr[offs+2*i+1];
  }
  r.arr[i0*r.s0+i1*r.s1+i2*r.s2+blockIdx.y*r.s3+threadIdx.x*r.s4]+=t;
  
}


__global__ void RtensorConvolve3d_sparse_kernel
(const cnine::Rtensor5_view r, const cnine::Rtensor5_view x, float* warr, int* wdir, 
  const int J0, const int J1, const int J2, const int padding0, const int padding1, const int padding2){

  int i0=blockIdx.x/(r.n1*r.n2);
  int i1=(blockIdx.x/r.n2)-i0*r.n1;
  int i2=blockIdx.x%r.n2;

  int offs=wdir[2*blockIdx.y];
  int n=wdir[2*blockIdx.y+1];
  int na=x.n3;

  float t=0;
  for(int i=0; i<n; i++){
    int s=*reinterpret_cast<int*>(warr+offs+2*i);
    int j0=s/(J1*J2*na);
    int j1=s/(J2*na)-j0*J1;
    int j2=s/na-(j0*J1*J2+j1*J2);
    int a=s%na;
    if(i0+j0-padding0<0 || i0+j0-padding0>=x.n0) continue;
    if(i1+j1-padding1<0 || i1+j1-padding1>=x.n1) continue;
    if(i2+j2-padding1<0 || i2+j2-padding2>=x.n2) continue;
    t+=x.arr[(i0+j0)*x.s0+(i1+j1)*x.s1+(i2+j2)*x.s2+a*x.s3+threadIdx.x*x.s4]*warr[offs+2*i+1];
  }
  r.arr[i0*r.s0+i1*r.s1+i2*r.s2+blockIdx.y*r.s3+threadIdx.x*r.s4]+=t;
  
}


// ---- 6D case (b,i0,i1,i2,a,c)*(a',j0,j1,j2,a) -> (b,i0+j0,i1+j1,i2+j2,a',c) -----------------------------------

__global__ void RtensorConvolve3d_sparse_kernel
(const cnine::Rtensor6_view r, const cnine::Rtensor6_view x, float* warr, int* wdir, 
  const int J0, const int J1, const int J2){

  int i0=blockIdx.x/(r.n2*r.n3);
  int i1=(blockIdx.x/r.n3)-i0*r.n2;
  int i2=blockIdx.x%r.n3;

  int offs=wdir[2*blockIdx.y];
  int n=wdir[2*blockIdx.y+1];
  int na=x.n4;

  float t=0;
  for(int i=0; i<n; i++){
    int s=*reinterpret_cast<int*>(warr+offs+2*i);
    int j0=s/(J1*J2*na);
    int j1=s/(J2*na)-j0*J1;
    int j2=s/na-(j0*J1*J2+j1*J2);
    int a=s%na;
    t+=x.arr[blockIdx.z*x.s0+(i0+j0)*x.s1+(i1+j1)*x.s2+(i2+j2)*x.s3+a*x.s4+threadIdx.x*x.s5]*warr[offs+2*i+1];
  }
  //printf("%d %d %d %d %d %d %f\n",blockIdx.z,i0,i1,i2,blockIdx.y,threadIdx.x,t);
  r.arr[blockIdx.z*r.s0+i0*r.s1+i1*r.s2+i2*r.s3+blockIdx.y*r.s4+threadIdx.x*r.s5]+=t;
  
}


__global__ void RtensorConvolve3d_sparse_kernel
(const cnine::Rtensor6_view r, const cnine::Rtensor6_view x, float* warr, int* wdir, 
  const int J0, const int J1, const int J2, const int padding0, const int padding1, const int padding2){

  int i0=blockIdx.x/(r.n2*r.n3);
  int i1=(blockIdx.x/r.n3)-i0*r.n2;
  int i2=blockIdx.x%r.n3;

  int offs=wdir[2*blockIdx.y];
  int n=wdir[2*blockIdx.y+1];
  int na=x.n4;

  float t=0;
  for(int i=0; i<n; i++){
    int s=*reinterpret_cast<int*>(warr+offs+2*i);
    int j0=s/(J1*J2*na);
    int j1=s/(J2*na)-j0*J1;
    int j2=s/na-(j0*J1*J2+j1*J2);
    int a=s%na;
    if(i0+j0-padding0<0 || i0+j0-padding0>=x.n1) continue;
    if(i1+j1-padding1<0 || i1+j1-padding1>=x.n2) continue;
    if(i2+j2-padding1<0 || i2+j2-padding2>=x.n3) continue;
    t+=x.arr[blockIdx.z*x.s0+(i0+j0)*x.s1+(i1+j1)*x.s2+(i2+j2)*x.s3+a*x.s4+threadIdx.x*x.s5]*warr[offs+2*i+1];
  }
  r.arr[blockIdx.z*r.s0+i0*r.s1+i1*r.s2+i2*r.s3+blockIdx.y*r.s4+threadIdx.x*r.s5]+=t;
  
}


// ----------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------


namespace cnine{


// ---- 4D case (i0,i1,i2,a)*(a',j0,j1,j2,a) -> (i0+j0,i1+j1,i2+j2,a') ---------------------------------------


  void RtensorConvolve3d_cu(const Rtensor4_view& r, const Rtensor4_view& x, const CSRmatrix<float>& w, 
    const int J0, const int J1, const int J2, 
    const int padding0, const int padding1,  const int padding2, const cudaStream_t& stream){
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(w.dev==1);

    dim3 blocks(r.n0*r.n1*r.n2);

    if(padding0==0&&padding1==0&&padding2==0)
      RtensorConvolve3d_sparse_kernel<<<blocks,r.n3,0,stream>>>(r,x,w.arrg,w.get_dirg(1),J0,J1,J2);
    else
      RtensorConvolve3d_sparse_kernel<<<blocks,r.n3,0,stream>>>(r,x,w.arrg,w.get_dirg(1),J0,J1,J2,padding0,padding1,padding2);

  }


// ---- 5D case (i0,i1,i2,a,c)*(a',j0,j1,j2,a) -> (i0+j0,i1+j1,i2+j2,a',c) -----------------------------------


  void RtensorConvolve3d_cu(const Rtensor5_view& r, const Rtensor5_view& x, const CSRmatrix<float>& w, 
    const int J0, const int J1, const int J2, 
    const int padding0, const int padding1,  const int padding2, const cudaStream_t& stream){
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(w.dev==1);

    dim3 blocks(r.n0*r.n1*r.n2,r.n3);

    if(padding0==0&&padding1==0&&padding2==0)
      RtensorConvolve3d_sparse_kernel<<<blocks,r.n4,0,stream>>>(r,x,w.arrg,w.get_dirg(1),J0,J1,J2);
    else
      RtensorConvolve3d_sparse_kernel<<<blocks,r.n4,0,stream>>>(r,x,w.arrg,w.get_dirg(1),J0,J1,J2,padding0,padding1,padding2);

  }


// ---- 6D case (b,i0,i1,i2,a,c)*(a',j0,j1,j2,a) -> (b,i0+j0,i1+j1,i2+j2,a',c) -----------------------------------


  void RtensorConvolve3d_cu(const Rtensor6_view& r, const Rtensor6_view& x, const CSRmatrix<float>& w, 
    const int J0, const int J1, const int J2, 
    const int padding0, const int padding1,  const int padding2, const cudaStream_t& stream){
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(w.dev==1);

    CNINE_ASSRT(x.n0==r.n0);
    CNINE_ASSRT(x.n5==r.n5);

    dim3 blocks(r.n1*r.n2*r.n3,r.n4,r.n0);

    if(padding0==0&&padding1==0&&padding2==0)
      RtensorConvolve3d_sparse_kernel<<<blocks,r.n5,0,stream>>>(r,x,w.arrg,w.get_dirg(1),J0,J1,J2);
    else
      RtensorConvolve3d_sparse_kernel<<<blocks,r.n5,0,stream>>>(r,x,w.arrg,w.get_dirg(1),J0,J1,J2,padding0,padding1,padding2);

  }

}


#endif 
