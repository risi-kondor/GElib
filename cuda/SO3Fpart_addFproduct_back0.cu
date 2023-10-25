/*
 * This file is part of GElib, a C++/CUDA library for group equivariant 
 * tensor operations. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in orginal
 * or modified form) must retain this copyright notice and must be 
 * accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _SO3Fpart_addFproduct_back0_cu
#define _SO3Fpart_addFproduct_back0_cu

#include <cuda.h>
#include <cuda_runtime.h>
//#include <thrust/complex.h>
//#include <thrust/tuple.h>

#include "SO3_CGbank.hpp"
#include "Ctensor3_view.hpp"


extern GElib::SO3_CGbank SO3_cgbank;


__device__ int loadg4(const cnine::Ctensor3_view& x, float* dest, const int b, const int t){
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


__device__ int saveg4(const cnine::Ctensor3_view& x, float* source, const int b, const int t){
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


__device__ int loadg4c(const cnine::Ctensor3_view& x, float* dest, const int b, const int t){
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

/*
__device__ int saveg4c(const cnine::Ctensor3_view& x, float* source, const int b, const int t){
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
*/


__global__ void SO3Fpart_addFproduct_back0_kernel(const cnine::Ctensor3_view r, const cnine::Ctensor3_view x, 
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
  float* xpi=xpr+loadg4(x,xpr,b,t);

  float* ypr=xpr+((2*xn*xn-1)/32+1)*32;
  float* ypi;
  if(conj==0) ypi=ypr+loadg4(y,ypr,b,t);
  else ypi=ypr+loadg4c(y,ypr,b,t);

  float* rpr=ypr+((2*yn*yn-1)/32+1)*32;
  float* rpi=rpr+loadg4(r,rpr,b,t);

  __syncthreads();

  if(t<xn*yn){

    int i1=t%xn;
    float* _xpr=xpr+i1;
    float* _xpi=xpi+i1;
    
    int i2=t/xn;
    ypr=ypr+i2;
    ypi=ypi+i2;
    
    int i=i1+i2-l1-l2+l;
    float* _rpr=rpr+i;
    float* _rpi=rpi+i;

    if(i>=0 && i<rn){
      float c0=cptr[i1*yn+i2]*xn*yn/rn;
      
      for(int m1=-l1; m1<=l1; m1++){
	int lower=-l-m1; if(lower<-l2) lower=-l2;
	int upper=l-m1; if(upper>l2) upper=l2;
	for(int m2=lower; m2<=upper; m2++){
	  float c=cptr[(m1+l1)*yn+m2+l2];
	  const float y_r=ypr[yn*(m2+l2)];
	  const float y_i=ypi[yn*(m2+l2)];
	  const float g_r=_rpr[rn*(m1+m2+l)];
	  const float g_i=_rpi[rn*(m1+m2+l)];
	  //_xpr[xn*(m1+l1)]+=c0*c*(g_r*y_r+g_i*y_i);
	  //_xpi[xn*(m1+l1)]+=c0*c*(-g_r*y_i+g_i*y_r);
	  atomicAdd(_xpr+xn*(m1+l1),c0*c*(g_r*y_r+g_i*y_i));
	  atomicAdd(_xpi+xn*(m1+l1),c0*c*(-g_r*y_i+g_i*y_r));
	}
 
      }
    }
  }

  __syncthreads();
  
  saveg4(x,xpr,b,t);

}


__global__ void SO3Fpart_addFproduct_back0_large_kernel(const cnine::Ctensor3_view r, const cnine::Ctensor3_view x, 
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
  float* xpi=xpr+loadg4(x,xpr,b,t);

  float* ypr=xpr+((2*xn*xn-1)/32+1)*32;
  float* ypi;
  if(conj==0) ypi=ypr+loadg4(y,ypr,b,t);
  else ypi=ypr+loadg4c(y,ypr,b,t);

  float* rpr=ypr+((2*yn*yn-1)/32+1)*32;
  float* rpi=rpr+loadg4(r,rpr,b,t);

  __syncthreads();

  if(t<xn){

    int i1=t;
    float* _xpr=xpr+i1;
    float* _xpi=xpi+i1;
    
    for(int i2=0; i2<yn; i2++){
      ypr=ypr+i2;
      ypi=ypi+i2;
    
      int i=i1+i2-l1-l2+l;
      float* _rpr=rpr+i;
      float* _rpi=rpi+i;

      if(i>=0 && i<rn){
	float c0=cptr[i1*yn+i2]*xn*yn/rn;
      
	for(int m1=-l1; m1<=l1; m1++){
	  int lower=-l-m1; if(lower<-l2) lower=-l2;
	  int upper=l-m1; if(upper>l2) upper=l2;
	  for(int m2=lower; m2<=upper; m2++){
	    float c=cptr[(m1+l1)*yn+m2+l2];
	    const float y_r=ypr[yn*(m2+l2)];
	    const float y_i=ypi[yn*(m2+l2)];
	    const float g_r=_rpr[rn*(m1+m2+l)];
	    const float g_i=_rpi[rn*(m1+m2+l)];
	    _xpr[xn*(m1+l1)]+=c0*c*(g_r*y_r+g_i*y_i);
	    _xpi[xn*(m1+l1)]+=c0*c*(-g_r*y_i+g_i*y_r);
	    //atomicAdd(_xpr+xn*(m1+l1),c0*c*(g_r*y_r+g_i*y_i));
	    //atomicAdd(_xpi+xn*(m1+l1),c0*c*(-g_r*y_i+g_i*y_r));
	  }
	}

      }
    }
  }

  __syncthreads();
  
  saveg4(x,xpr,b,t);

}


namespace GElib{


  void SO3Fpart_addFproduct_back0_cu(const cnine::Ctensor3_view& x, const cnine::Ctensor3_view& r, const cnine::Ctensor3_view& y, 
    const int conj, const int method, const cudaStream_t& stream){

    const int xl=(x.n1-1)/2;
    const int yl=(y.n1-1)/2;
    const int l=(r.n1-1)/2;

    const int b=r.n0;
    assert(x.n0==b);
    assert(y.n0==b);

    float* cptr=nullptr;
    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    if(Cptr<0) cptr=SO3_cgbank.getf(CGindex(xl,yl,l),r.dev).arrg;
    int clines=cnine::roundup(x.n1*y.n1,32)/32;

    int nlines=cnine::roundup(x.n1*x.n2*2,32)/32+
      cnine::roundup(y.n1*y.n2*2,32)/32+
      cnine::roundup(r.n1*r.n2*2,32)/32;


    if(nlines<=384){

      if(method==0){

      SO3Fpart_addFproduct_back0_kernel<<<b,cnine::roundup(x.n2*y.n2,32),nlines*128,stream>>>
	(r,x,y,Cptr,cptr,conj);

      }else{

	SO3Fpart_addFproduct_back0_large_kernel<<<b,cnine::roundup(std::max(std::max(x.n2,y.n2),r.n2),32),nlines*128,stream>>>
	  (r,x,y,Cptr,cptr,conj);

      }

    }else{
      cout<<"error"<<endl;
    }

  }    


}


#endif 

