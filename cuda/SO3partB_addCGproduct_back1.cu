// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022s, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3partB_addCGproduct_back1_cu
#define _SO3partB_addCGproduct_back1_cu

#include <cuda.h>
#include <cuda_runtime.h>
//#include <thrust/complex.h>
//#include <thrust/tuple.h>

#include "SO3_CGbank.hpp"
#include "Ctensor3_view.hpp"

//__device__ __constant__ unsigned char cg_cmem[32276]; 

extern GElib::SO3_CGbank SO3_cgbank;


__device__ int loadg2(const cnine::Ctensor3_view& x, float* dest, const int b, const int t){
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


__device__ int saveg2(const cnine::Ctensor3_view& x, float* source, const int b, const int t){
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


__global__ void SO3partB_addCGproduct_back1_kernel(const cnine::Ctensor3_view y, const cnine::Ctensor3_view r, 
  const cnine::Ctensor3_view x, const int Cptr){

  extern __shared__ unsigned char _shared[]; 
  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int b=blockIdx.x;
  const int t=threadIdx.x;

//printf("%d",t);

  int l1=(x.n1-1)/2;
  int l2=(y.n1-1)/2;
  int l=(r.n1-1)/2;
  int xn=x.n2;
  int yn=y.n2;
  int rn=xn*yn;
  int L2=y.n1;

//for(int i=0; i<x.n1; i++)
//for(int j=0; j<y.n1; j++)
//if(t==0)printf("%f\n",C_ptr[i*L2+j]);

  float* xpr=reinterpret_cast<float*>(_shared);
  float* xpi=xpr+loadg2(x,xpr,b,t);

  float* ypr=xpr+((2*x.n1*xn-1)/32+1)*32;
  float* ypi=ypr+loadg2(y,ypr,b,t);

  float* rpr=ypr+((2*y.n1*yn-1)/32+1)*32;
  float* rpi=rpr+loadg2(r,rpr,b,t);

  __syncthreads();



  for(int xcol=0; xcol<xn; xcol++){
    if(t<yn){

      float* _xpr=xpr+xcol;
      float* _xpi=xpi+xcol;

      float* _ypr=ypr+t;
      float* _ypi=ypi+t;
      
      float* _rpr=rpr+xcol*yn+t;
      float* _rpi=rpi+xcol*yn+t;
      
      for(int m1=-l1; m1<=l1; m1++){
	const float x_r=_xpr[xn*(m1+l1)];
	const float x_i=_xpi[xn*(m1+l1)];
	int lower=-l-m1; if(lower<-l2) lower=-l2;
	int upper=l-m1; if(upper>l2) upper=l2;
	for(int m2=lower; m2<=upper; m2++){
	  float c=C_ptr[(m1+l1)*L2+m2+l2];
	  const float g_r=_rpr[rn*(m1+m2+l)];
	  const float g_i=_rpi[rn*(m1+m2+l)];
	  _ypr[yn*(m2+l2)]+=c*(g_r*x_r+g_i*x_i);
	  _ypi[yn*(m2+l2)]+=c*(-g_r*x_i+g_i*x_r);
	}
      }
    }
    __syncthreads();
  }
  

  __syncthreads();
  
  saveg2(y,ypr,b,t);

}



namespace GElib{


  void SO3partB_addCGproduct_back1_cu(const cnine::Ctensor3_view& yg, cnine::Ctensor3_view g, const cnine::Ctensor3_view& x, 
    const int offs, const cudaStream_t& stream){

    const int xl=(x.n1-1)/2;
    const int yl=(yg.n1-1)/2;
    const int l=(g.n1-1)/2;

    const int b=g.n0;
    assert(x.n0==b);
    assert(yg.n0==b);

    g.arr+=g.s2*offs;
    g.arrc+=g.s2*offs;
    g.n2=x.n2*yg.n2;

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;

    int nlines=cnine::roundup(x.n1*x.n2*2,32)/32+
      cnine::roundup(yg.n1*yg.n2*2,32)/32+
      cnine::roundup(g.n1*x.n2*yg.n2*2,32)/32;

    if(nlines<=384){

      SO3partB_addCGproduct_back1_kernel<<<b,cnine::roundup(x.n2*yg.n2,32),nlines*128,stream>>>
	(yg,g,x,Cptr);

    }else{
      cout<<"error"<<endl;
    }

    //r.arr-=r.s1*offs;
    //r.arrc-=r.s1*offs;
    //r.n1=rn1;

  }    


}


#endif 


