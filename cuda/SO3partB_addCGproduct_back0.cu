// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022s, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3partB_addCGproduct_back0_cu
#define _SO3partB_addCGproduct_back0_cu

#include <cuda.h>
#include <cuda_runtime.h>
//#include <thrust/complex.h>
//#include <thrust/tuple.h>

#include "SO3_CGbank.hpp"
#include "Ctensor3_view.hpp"

//__device__ __constant__ unsigned char cg_cmem[32276]; 

extern GElib::SO3_CGbank SO3_cgbank;


__device__ int loadg1(const cnine::Ctensor3_view& x, float* dest, const int b, const int t){
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


__device__ int saveg1(const cnine::Ctensor3_view& x, float* source, const int b, const int t){
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


__global__ void SO3partB_addCGproduct_back0_kernel(const cnine::Ctensor3_view x, const cnine::Ctensor3_view r, 
  const cnine::Ctensor3_view y, const int Cptr){

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
  float* xpi=xpr+loadg1(x,xpr,b,t);

  float* ypr=xpr+((2*x.n1*xn-1)/32+1)*32;
  float* ypi=ypr+loadg1(y,ypr,b,t);

  float* rpr=ypr+((2*y.n1*yn-1)/32+1)*32;
  float* rpi=rpr+loadg1(r,rpr,b,t);

  __syncthreads();


  float* _xpr=xpr+t;
  float* _xpi=xpi+t;

  for(int ycol=0; ycol<yn; ycol++){
if(t<xn){

      float* _ypr=ypr+ycol;
      float* _ypi=ypi+ycol;
      
      float* _rpr=rpr+t*yn+ycol;
      float* _rpi=rpi+t*yn+ycol;
      
      for(int m1=-l1; m1<=l1; m1++){
	int lower=-l-m1; if(lower<-l2) lower=-l2;
	int upper=l-m1; if(upper>l2) upper=l2;
	for(int m2=lower; m2<=upper; m2++){
	  float c=C_ptr[(m1+l1)*L2+m2+l2];
	  const float y_r=_ypr[yn*(m2+l2)];
	  const float y_i=_ypi[yn*(m2+l2)];
	  const float g_r=_rpr[rn*(m1+m2+l)];
	  const float g_i=_rpi[rn*(m1+m2+l)];
    //if(t==0) printf("%f %f %f %d\n",y_r,g_r,c,(m1+l1)*L2+m2+l2);
	  _xpr[xn*(m1+l1)]+=c*(g_r*y_r+g_i*y_i);
	  _xpi[xn*(m1+l1)]+=c*(-g_r*y_i+g_i*y_r);
  }
      }
}
      __syncthreads();
  }
  

  __syncthreads();
  
  saveg1(x,xpr,b,t);

}



namespace GElib{


  void SO3partB_addCGproduct_back0_cu(const cnine::Ctensor3_view& xg, cnine::Ctensor3_view rg, const cnine::Ctensor3_view& y, 
    const int offs, const cudaStream_t& stream){

    const int xl=(xg.n1-1)/2;
    const int yl=(y.n1-1)/2;
    const int l=(rg.n1-1)/2;

    const int b=rg.n0;
    assert(xg.n0==b);
    assert(y.n0==b);

    rg.arr+=rg.s2*offs;
    rg.arrc+=rg.s2*offs;
    rg.n2=xg.n2*y.n2;

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;

    int nlines=cnine::roundup(xg.n1*xg.n2*2,32)/32+
      cnine::roundup(y.n1*y.n2*2,32)/32+
      cnine::roundup(rg.n1*xg.n2*y.n2*2,32)/32;

    if(nlines<=384){

      SO3partB_addCGproduct_back0_kernel<<<b,cnine::roundup(xg.n2*y.n2,32),nlines*128,stream>>>
	(xg,rg,y,Cptr);

    }else{
      cout<<"error"<<endl;
    }

    //r.arr-=r.s1*offs;
    //r.arrc-=r.s1*offs;
    //r.n1=rn1;

  }    


}


#endif 



  /*
  if(t<32){
    int xn=xview.n1;
    int xs0=xview.s0;
    int xs1=xview.s1;
    int xarr=xview.arr;
    int xarrc=xview.arrc;
    for(int i=0; i<2*l1+1; i++)
      for(int j=0; j<xn; x++)
	xpr[i*xwidth+j]=xarr[i*xs0+j*xs1];
    for(int i=0; i<2*l1+1; i++)
      for(int j=0; j<xn; x++)
	xpi[i*xwidth+j]=xarrc[i*xs0+j*xs1];
  }

  if(t<32){
    int yn=yview.n1;
    int ys0=yview.s0;
    int ys1=yview.s1;
    int yarr=yview.arr;
    int yarrc=yview.arrc;
    for(int i=0; i<2*l2+1; i++)
      for(int j=0; j<xn; x++)
	ypr[i*ywidth+j]=yarr[i*ys0+j*ys1];
    for(int i=0; i<2*l2+1; i++)
      for(int j=0; j<xn; x++)
	ypi[i*ywidth+j]=yarrc[i*ys0+j*ys1];
  }

  if(t<rwidth){
    for(int m1=-l1; m1<=l1; m1++){
      const float x_r=xpr[xwidth*(m1+l1)];
      const float x_i=xpi[xwidth*(m1+l1)];
      int lower=-l-m1; if(lower<-l2) lower=-l2;
      int upper=l-m1; if(upper>l2) upper=l2;
      for(int m2=lower; m2<=upper; m2++){
	float c=C_ptr[(m1+l1)*r2+m2+l2];
	const float y_r=shared[ypr+ywidth*(m2+l2)];
	const float y_i=shared[ypi+ywidth*(m2+l2)];
	shared[rpr+rwidth*(m1+m2+l)]+=c*(x_r*y_r-x_i*y_i); 
	shared[rpi+rwidth*(m1+m2+l)]+=c*(x_r*y_i+x_i*y_r);
      }
    }
  }
  */
