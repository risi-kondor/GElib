// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022s, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3Fpart_addFproduct_back1_cu
#define _SO3Fpart_addFproduct_back1_cu

#include <cuda.h>
#include <cuda_runtime.h>
//#include <thrust/complex.h>
//#include <thrust/tuple.h>

#include "SO3_CGbank.hpp"
#include "Ctensor3_view.hpp"


extern GElib::SO3_CGbank SO3_cgbank;


__device__ int loadg5(const cnine::Ctensor3_view& x, float* dest, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* destc=dest+offs;
  float* source=x.arr+x.s0*b;
  float* sourcec=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*J+t]=source[i*s1+t*s2];
    for(int i=0; i<I; i++)
      destc[i*J+t]=sourcec[i*s1+t*s2];
  }
  return offs;
}


__device__ int saveg5(const cnine::Ctensor3_view& x, float* source, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
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


__device__ int loadg5c(const cnine::Ctensor3_view& x, float* dest, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* destc=dest+offs;
  float* source=x.arr+x.s0*b;
  float* sourcec=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*J+t]=source[i*s1+t*s2];
    for(int i=0; i<I; i++)
      destc[i*J+t]=-sourcec[i*s1+t*s2];
  }
  return offs;
}


__device__ int saveg5c(const cnine::Ctensor3_view& x, float* source, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* sourcec=source+offs;
  float* dest=x.arr+x.s0*b;
  float* destc=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*s1+t*s2]=source[i*J+t];
    for(int i=0; i<I; i++)
      destc[i*s1+t*s2]=-sourcec[i*J+t];
  }
  return offs;
}



__global__ void SO3Fpart_addFproduct_back1_kernel(const cnine::Ctensor3_view r, const cnine::Ctensor3_view x, 
  const cnine::Ctensor3_view y, const int Cptr, float* cptr_global, const int conj){

  extern __shared__ unsigned char _shared[]; 
  const int b=blockIdx.x;
  const int t=threadIdx.x;

  int l1=(x.n1-1)/2;
  int l2=(y.n1-1)/2;
  int l=(r.n1-1)/2;
  int xn=x.n2;
  int yn=y.n2;
  int rn=r.n2;

  float* cptr;
  if(Cptr>=0) cptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  else cptr=cptr_global;

  float* xpr=reinterpret_cast<float*>(_shared);
  float* xpi=xpr+loadg5(x,xpr,b,t);

  float* ypr=xpr+((2*xn*xn-1)/32+1)*32;
  float* ypi;
  if(conj==0) ypi=ypr+loadg5(y,ypr,b,t);
  else ypi=ypr+loadg5c(y,ypr,b,t);

  float* rpr=ypr+((2*yn*yn-1)/32+1)*32;
  float* rpi=rpr+loadg5(r,rpr,b,t);

  __syncthreads();

  if(t<xn*yn){

    int i1=t/yn;
    float* _xpr=xpr+i1;
    float* _xpi=xpi+i1;
    
    int i2=t%yn;
    float* _ypr=ypr+i2;
    float* _ypi=ypi+i2;
    
    int i=i1+i2-l1-l2+l;
    float* _rpr=rpr+i;
    float* _rpi=rpi+i;

    if(i>=0 && i<rn){
      float c0=cptr[i1*yn+i2]*xn*yn/rn;
      
      for(int m1=-l1; m1<=l1; m1++){
	const float x_r=_xpr[xn*(m1+l1)];
	const float x_i=_xpi[xn*(m1+l1)];
	int lower=-l-m1; if(lower<-l2) lower=-l2;
	int upper=l-m1; if(upper>l2) upper=l2;
	for(int m2=lower; m2<=upper; m2++){
	  float c=cptr[(m1+l1)*yn+m2+l2];
	  const float g_r=_rpr[rn*(m1+m2+l)];
	  const float g_i=_rpi[rn*(m1+m2+l)];
	  //_ypr[yn*(m2+l2)]+=c*(g_r*x_r+g_i*x_i);
	  //_ypi[yn*(m2+l2)]+=c*(-g_r*x_i+g_i*x_r);
	  atomicAdd(_ypr+yn*(m2+l2),c0*c*(g_r*x_r+g_i*x_i));
	  atomicAdd(_ypi+yn*(m2+l2),c0*c*(-g_r*x_i+g_i*x_r));
	}
 
      }
    }
  }

  __syncthreads();
  
  if(conj==0) saveg5(y,ypr,b,t);
  else saveg5c(y,ypr,b,t);

}



namespace GElib{


  void SO3Fpart_addFproduct_back1_cu(const cnine::Ctensor3_view& y, const cnine::Ctensor3_view& g, const cnine::Ctensor3_view& x, 
    const int conj, const cudaStream_t& stream){

    const int xl=(x.n1-1)/2;
    const int yl=(y.n1-1)/2;
    const int l=(g.n1-1)/2;

    const int b=g.n0;
    assert(x.n0==b);
    assert(y.n0==b);

    float* cptr=nullptr;
    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    if(Cptr<0) cptr=SO3_cgbank.getf(CGindex(xl,yl,l),g.dev).arrg;
    int clines=cnine::roundup(x.n1*y.n1,32)/32;

    int nlines=cnine::roundup(x.n1*x.n2*2,32)/32+
      cnine::roundup(y.n1*y.n2*2,32)/32+
      cnine::roundup(g.n1*g.n2*2,32)/32;


    if(nlines<=384){

      SO3Fpart_addFproduct_back1_kernel<<<b,cnine::roundup(x.n2*y.n2,32),nlines*128,stream>>>
	(g,x,y,Cptr,cptr,conj);

    }else{
      cout<<"error"<<endl;
    }

  }    


}


#endif 

