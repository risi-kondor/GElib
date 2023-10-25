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

#ifndef _SO3Fpart_addFproduct_cu
#define _SO3Fpart_addFproduct_cu

#include <cuda.h>
#include <cuda_runtime.h>
//#include <thrust/complex.h>
//#include <thrust/tuple.h>

#include "SO3_CGbank.hpp"
#include "Ctensor2_view.hpp"
#include "Ctensor3_view.hpp"

//__device__ __constant__ unsigned char cg_cmem[32276]; 

extern GElib::SO3_CGbank SO3_cgbank;




__device__ int loadg3(const cnine::Ctensor3_view& x, float* dest, const int b, const int t){
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


__device__ int saveg3(const cnine::Ctensor3_view& x, float* source, const int b, const int t){
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

__device__ int loadg3c(const cnine::Ctensor3_view& x, float* dest, const int b, const int t){
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
__device__ int saveg3c(const cnine::Ctensor3_view& x, float* source, const int b, const int t){
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


__global__ void SO3Fpart_addFproduct_kernel(const cnine::Ctensor3_view r, const cnine::Ctensor3_view x, 
  const cnine::Ctensor3_view y, const int Cptr, float* cptr_global, const int conj){

  extern __shared__ unsigned char _shared[]; 
  //const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
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
  float* xpi=xpr+x.n1*x.n2;
  loadg3(x,xpr,b,t);

  float* ypr=xpr+((2*xn*xn-1)/32+1)*32;
  float* ypi=ypr+y.n1*y.n2;
  if(conj==0) loadg3(y,ypr,b,t);
  else loadg3c(y,ypr,b,t);

  float* rpr=ypr+((2*yn*yn-1)/32+1)*32;
  float* rpi=rpr+r.n1*r.n2;
  loadg3(r,rpr,b,t);

  __syncthreads();

  if(t<xn*yn){

    int i1=t/yn;
    xpr=xpr+i1;
    xpi=xpi+i1;
    
    int i2=t%yn;
    ypr=ypr+i2;
    ypi=ypi+i2;
    
    int i=i1+i2-l1-l2+l;
    float* _rpr=rpr+i;
    float* _rpi=rpi+i;

    if(i>=0 && i<rn){

      float c0=cptr[i1*yn+i2]*xn*yn/rn;
      
      for(int m1=-l1; m1<=l1; m1++){
	const float x_r=xpr[xn*(m1+l1)];
	const float x_i=xpi[xn*(m1+l1)];
	int lower=-l-m1; if(lower<-l2) lower=-l2;
	int upper=l-m1; if(upper>l2) upper=l2;
	for(int m2=lower; m2<=upper; m2++){
	  float c=cptr[(m1+l1)*yn+m2+l2];
	  const float y_r=ypr[yn*(m2+l2)];
	  const float y_i=ypi[yn*(m2+l2)];
	  //_rpr[rn*(m1+m2+l)]+=c0*c*(x_r*y_r-x_i*y_i); 
	  //_rpi[rn*(m1+m2+l)]+=c0*c*(x_r*y_i+x_i*y_r);
	  atomicAdd(_rpr+rn*(m1+m2+l),c0*c*(x_r*y_r-x_i*y_i)); 
	  atomicAdd(_rpi+rn*(m1+m2+l),c0*c*(x_r*y_i+x_i*y_r));
	}
 
      }
    }
  }

  __syncthreads();
  
  saveg3(r,rpr,b,t);

}


__global__ void SO3Fpart_addFproduct_large_kernel(const cnine::Ctensor3_view r, const cnine::Ctensor3_view x, 
  const cnine::Ctensor3_view y, const int Cptr, float* cptr_global, const int conj){

  extern __shared__ unsigned char _shared[]; 
  //const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
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
  float* xpi=xpr+x.n1*x.n2;
  loadg3(x,xpr,b,t);

  float* ypr=xpr+((2*xn*xn-1)/32+1)*32;
  float* ypi=ypr+y.n1*y.n2;
  if(conj==0) loadg3(y,ypr,b,t);
  else loadg3c(y,ypr,b,t);

  float* rpr=ypr+((2*yn*yn-1)/32+1)*32;
  float* rpi=rpr+r.n1*r.n2;
  loadg3(r,rpr,b,t);

  //int tn=xn*yn;
  //float* tpr=rpr+((2*rn*rn-1)/32+1)*32;
  //float* tpi=tpr+tn*rn;

  __syncthreads();

  if(t<rn){
    int i=t;
    float* _rpr=rpr+i+rn*l;
    float* _rpi=rpi+i+rn*l;

    for(int i1=max(0,i-l-l2+l1); i1<min(xn,i-l+l2+l1+1); i1++){
      float* _xpr=xpr+i1;
      float* _xpi=xpi+i1;

      int i2=i-l-(i1-l1)+l2;
      float* _ypr=ypr+i2+yn*l2;
      float* _ypi=ypi+i2+yn*l2;

      float c0=cptr[i1*yn+i2]*xn*yn/rn;
      
      for(int m1=-l1; m1<=l1; m1++){
	const float x_r=_xpr[xn*(m1+l1)];
	const float x_i=_xpi[xn*(m1+l1)];
	int lower=-l-m1; if(lower<-l2) lower=-l2;
	int upper=l-m1; if(upper>l2) upper=l2;

	for(int m2=lower; m2<=upper; m2++){
	  float c=cptr[(m1+l1)*yn+m2+l2];
	  const float y_r=_ypr[yn*(m2)];
	  const float y_i=_ypi[yn*(m2)];
	  _rpr[rn*(m1+m2)]+=c0*c*(x_r*y_r-x_i*y_i); 
	  _rpi[rn*(m1+m2)]+=c0*c*(x_r*y_i+x_i*y_r);
	  //atomicAdd(_rpr+rn*(m1+m2+l),c0*c*(x_r*y_r-x_i*y_i)); 
	  //atomicAdd(_rpi+rn*(m1+m2+l),c0*c*(x_r*y_i+x_i*y_r));
	}
 
      }

    }
  }

  __syncthreads();
  
  saveg3(r,rpr,b,t);

}


namespace GElib{


  void SO3Fpart_addFproduct_cu(const cnine::Ctensor3_view& r, const cnine::Ctensor3_view& x, const cnine::Ctensor3_view& y, 
    const int conj, const int method, const cudaStream_t& stream){

    const int xl=(x.n1-1)/2;
    const int yl=(y.n1-1)/2;
    const int l=(r.n1-1)/2;
    const int b=r.n0;

    float* cptr=nullptr;
    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    if(Cptr<0) cptr=SO3_cgbank.getf(CGindex(xl,yl,l),r.dev).arrg;
    int clines=cnine::roundup(x.n1*y.n1,32)/32;

    int nlines=cnine::roundup(x.n1*x.n2*2,32)/32+
      cnine::roundup(y.n1*y.n2*2,32)/32+
      cnine::roundup(r.n1*r.n2*2,32)/32;

    if(nlines<=384){

      if(method==0){

	SO3Fpart_addFproduct_kernel<<<b,cnine::roundup(x.n2*y.n2,32),nlines*128,stream>>>
	  (r,x,y,Cptr,cptr,conj);
	return; 

      }else{

	SO3Fpart_addFproduct_large_kernel<<<b,cnine::roundup(std::max(std::max(x.n2,y.n2),r.n2),32),nlines*128,stream>>>
	  (r,x,y,Cptr,cptr,conj);
	return; 

      }
    }

    cout<<"error"<<endl;

  }    


}


#endif 

