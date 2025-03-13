/*
 * This file is part of GElib, a C++/CUDA library for group equivariant 
 * tensor operations. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in orginal
 * or modified form) must retain this copyright notice and must be 
 * accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _GElib_cuda_utils
#define _GElib_cuda_utils

#include <cuda.h>
#include <cuda_runtime.h>
#include "Ctensor5_view.hpp"

#define tix threadIdx.x


__forceinline__ __device__ void loadf(float* dest, const float* src, const int n){
  int nthreads=blockDim.x;
  int I=n/nthreads;
  for(int i=0; i<I; i++)
    dest[i*nthreads+tix]=src[i*nthreads+tix];
  if(tix<n-I*nthreads)
    dest[I*nthreads+tix]=src[I*nthreads+tix];
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


// Load n fragments from x to dest 
// assumption: number of threads is at least n
__forceinline__ __device__ int loadg_tile(float* dest, const cnine::Ctensor5_view& x, const int i, const int n){
  int I=x.n2;
  int J=x.n4;
  int s2=x.s2;
  int s4=x.s4;
  float* destc=dest+I*J;
  float* source=x.arr+x.s0*blockIdx.x+x.s1*blockIdx.y+i*x.s3;
  float* sourcec=x.arrc+x.s0*blockIdx.x+x.s1*blockIdx.y+i*x.s3;
  if(tix<n){
    for(int i=0; i<I; i++)
      dest[i*J+tix]=source[i*s2+tix*s4];
    for(int i=0; i<I; i++)
      destc[i*J+tix]=sourcec[i*s2+tix*s4];
  }
  return I*J;
}


// Save n fragments from src to x  
// assumption: number of threads is at least n
__forceinline__ __device__ void saveg_tile(float* src, const cnine::Ctensor5_view& x, const int i, const int n){
  int I=x.n2;
  int J=x.n4;
  int s2=x.s2;
  int s4=x.s4;
  float* srcc=src+I*J;
  float* dest=x.arr+x.s0*blockIdx.x+x.s1*blockIdx.y+i*x.s3;
  float* destc=x.arrc+x.s0*blockIdx.x+x.s1*blockIdx.y+i*x.s3; 
    if(tix<n){
    for(int i=0; i<I; i++)
      dest[i*s2+tix*s4]=src[i*J+tix];
    for(int i=0; i<I; i++)
      destc[i*s2+tix*s4]=srcc[i*J+tix];
  }
}


#endif 
