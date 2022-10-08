// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElib_cuda_loaders
#define _GElib_cuda_loaders

#include <cuda.h>
#include <cuda_runtime.h>
#include "Ctensor3_view.hpp"
#include "Ctensor4_view.hpp"

#define tix threadIdx.x

/*
__forceinline__ __device__ unsigned dynamic_smem_size(){
    unsigned ret; 
    asm volatile ("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
    return ret;
}
*/

__forceinline__ __device__ void loadf(float* dest, const float* src, const int n, const int t){
  int nthreads=blockDim.x;
  int I=n/nthreads;
  for(int i=0; i<I; i++)
    dest[i*nthreads+t]=src[i*nthreads+t];
  if(t<n-I*nthreads)
    dest[I*nthreads+t]=src[I*nthreads+t];
}


__forceinline__ __device__ void loadf(float* dest, const float* src, const int n){
  int nthreads=blockDim.x;
  int I=n/nthreads;
  for(int i=0; i<I; i++)
    dest[i*nthreads+tix]=src[i*nthreads+tix];
  if(tix<n-I*nthreads)
    dest[I*nthreads+tix]=src[I*nthreads+tix];
}


__forceinline__ __device__ int loadg(const cnine::Ctensor3_view& x, float* dest, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  float* destc=dest+I*J;
  float* source=x.arr+x.s0*b;
  float* sourcec=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*J+t]=source[i*s1+t*s2];
    for(int i=0; i<I; i++)
      destc[i*J+t]=sourcec[i*s1+t*s2];
  }
  return I*J;
}


// Load n fragments from x to dest 
// assumption: number of threads is at least n
__forceinline__ __device__ int loadg_tile(float* dest, const cnine::Ctensor4_view& x, const int b, const int i, const int n){
  int I=x.n1;
  int J=x.n3;
  int s1=x.s1;
  int s3=x.s3;
  float* destc=dest+I*J;
  float* source=x.arr+x.s0*b+i*x.s2;
  float* sourcec=x.arrc+x.s0*b+i*x.s2;
  if(tix<n){
    for(int i=0; i<I; i++)
      dest[i*J+tix]=source[i*s1+tix*s3];
    for(int i=0; i<I; i++)
      destc[i*J+tix]=sourcec[i*s1+tix*s3];
  }
  return I*J;
}


// Save n fragments from to x  
// assumption: number of threads is at least n
__forceinline__ __device__ void saveg_tile(float* src, const cnine::Ctensor4_view& x, const int b, const int i, const int n){
  int I=x.n1;
  int J=x.n3;
  int s1=x.s1;
  int s3=x.s3;
  float* srcc=src+I*J;
  float* dest=x.arr+x.s0*b+i*x.s2;
  float* destc=x.arrc+x.s0*b+i*x.s2;
  if(tix<n){
    for(int i=0; i<I; i++)
      dest[i*s1+tix*s3]=src[i*J+tix];
    for(int i=0; i<I; i++)
      destc[i*s1+tix*s3]=srcc[i*J+tix];
  }
}


__forceinline__ __device__ int saveg(const cnine::Ctensor3_view& x, float* source, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J;
  float* sourcec=source+offs;
  float* dest=x.arr+x.s0*b;
  float* destc=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*s1+t*s2]=source[i*J+t];
    for(int i=0; i<I; i++)
      destc[i*s1+t*s2]=sourcec[i*J+t];
  }
  return offs;
}


#undef tix 

#endif
