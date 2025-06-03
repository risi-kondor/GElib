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


__global__ void RtensorConvolve2d_sparse_kernel
(float* rarr, const int rs0, const int rs1, const int rs2, const int rs3, const int rs4, 
  float* xarr, const int xs0, const int xs1, const int xs2, const int xs3, const int xs4,  
  float* warr, int* wdir, const int rn1, const int nj1, const int na){

  int i0=blockIdx.y/rn1;
  int i1=blockIdx.y%rn1;

  int row=blockIdx.y*blockDim.z+blockIdx.z;
  int offs=wdir[2*row];
  int n=wdir[2*row+1];
  
  float t=0;
  for(int i=0; i<n; i++){
    int s=*reinterpret_cast<int*>(warr+offs+2*i);
    int j0=s/(nj1*na);
    int j1=(s/na)%nj1;
    int a=s%na;
    t+=xarr[blockIdx.x*xs0+(i0+j0)*xs1+(i1+j1)*xs2+a*xs3+threadIdx.x*xs4]*warr[offs+2*i+1];
  }
  rarr[blockIdx.x*rs0+i0*rs1+i1*rs2+blockIdx.z*rs3+threadIdx.x*rs4]+=t;
  
}


__global__ void RtensorConvolve2d_sparse_kernel
(float* rarr, const int rs0, const int rs1, const int rs2, const int rs3, const int rs4, 
  float* xarr, const int xs0, const int xs1, const int xs2, const int xs3, const int xs4,  
  float* warr, int* wdir, const int rn1, const int nj1, const int na,
  const int xn1, const int xn2, const int padding0, const int padding1){

  int i0=blockIdx.y/rn1;
  int i1=blockIdx.y%rn1;

  int row=blockIdx.y*blockDim.z+blockIdx.z;
  int offs=wdir[2*row];
  int n=wdir[2*row+1];
  
  float t=0;
  for(int i=0; i<n; i++){
    int s=*reinterpret_cast<int*>(warr+offs+2*i);
    int j0=s/(nj1*na);
    int j1=(s/na)%nj1;
    if(i0+j0-padding0<0 || i0+j0-padding0>=xn1) continue;
    if(i1+j1-padding1<0 || i1+j1-padding1>=xn2) continue;
    int a=s%na;
    t+=xarr[blockIdx.x*xs0+(i0+j0-padding0)*xs1+(i1+j1-padding1)*xs2+a*xs3+threadIdx.x*xs4]*warr[offs+2*i+1];
  }
  rarr[blockIdx.x*rs0+i0*rs1+i1*rs2+blockIdx.z*rs3+threadIdx.x*rs4]+=t;
  
}


// ----------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------


namespace cnine{


  void RtensorConvolve2d_cu(const Rtensor5_view& r, const Rtensor5_view& x, const CSRmatrix<float>& w, 
    const int J0, const int J1, const int padding0, const int padding1, const cudaStream_t& stream){
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(w.dev==1);

    dim3 blocks(r.n0,r.n1*r.n2,r.n3);

    if(padding0==0&&padding1==0){
      RtensorConvolve2d_sparse_kernel<<<blocks,r.n4,0,stream>>>
	(r.arr,r.s0,r.s1,r.s2,r.s3,r.s4,
	  x.arr,x.s0,x.s1,x.s2,x.s3,x.s4,
	  w.arrg,w.get_dirg(1),
	  J0,J1,w.m); // check this
    }else{
      //RtensorConvolve2d_sparse_kernel<<<blocks,r.n4,0,stream>>>
      //(r.arr,r.s0,r.s1,r.s2,r.s3,r.s4,
      //  x.arr,x.s0,x.s1,x.s2,x.s3,x.s4,
      //  w.arrg,w.get_dirg(1),
      //  r.n1,w.n2,w.m,
      //  x.n1,x.n2,padding0,padding1); 
    }

  }

}


#endif 
