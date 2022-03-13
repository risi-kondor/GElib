// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3partB_addCGproduct_back0_cu
#define _SO3partB_addCGproduct_back0_cu

#include <cuda.h>
#include <cuda_runtime.h>

#include "SO3_CGbank.hpp"
#include "Ctensor3_view.hpp"
#include "cuda_loaders.cu"


extern GElib::SO3_CGbank SO3_cgbank;




__global__ void SO3partB_addCGproduct_back0_kernel(const cnine::Ctensor3_view x, const cnine::Ctensor3_view r, 
  const cnine::Ctensor3_view y, const int Cptr){

  extern __shared__ unsigned char _shared[]; 
  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int b=blockIdx.x;
  const int t=threadIdx.x;

  int l1=(x.n1-1)/2;
  int l2=(y.n1-1)/2;
  int l=(r.n1-1)/2;
  int xn=x.n2;
  int yn=y.n2;
  int rn=xn*yn;
  int L2=y.n1;

  float* xpr=reinterpret_cast<float*>(_shared);
  float* xpi=xpr+loadg(x,xpr,b,t);

  float* ypr=xpr+((2*x.n1*xn-1)/32+1)*32;
  float* ypi=ypr+loadg(y,ypr,b,t);

  float* rpr=ypr+((2*y.n1*yn-1)/32+1)*32;
  float* rpi=rpr+loadg(r,rpr,b,t);

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
	  _xpr[xn*(m1+l1)]+=c*(g_r*y_r+g_i*y_i);
	  _xpi[xn*(m1+l1)]+=c*(-g_r*y_i+g_i*y_r);
	}
      }
    }
    __syncthreads();
  }
  

  __syncthreads();
  
  saveg(x,xpr,b,t);

}


__global__ void SO3partB_addCGproduct_back0_tiled_kernel(const cnine::Ctensor4_view_t3 x, const cnine::Ctensor3_view r, 
  const cnine::Ctensor4_view_t3 y, const int Cptr, const bool preloadCG){

  extern __shared__ unsigned char _shared[]; 
  const int b=blockIdx.x;
  const int t=threadIdx.x;

  int l1=(x.n1-1)/2;
  int l2=(y.n1-1)/2;
  int l=(r.n1-1)/2;
  int L2=y.n1;

  float* cptr;
  float* xpr;
  if(preloadCG){
    cptr=reinterpret_cast<float*>(_shared);
    xpr=cptr+((x.n1*y.n1-1)/32+1)*32;
    loadf(cptr,reinterpret_cast<float*>(cg_cmem)+Cptr,x.n1*y.n1);
  }else{
    cptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
    xpr=reinterpret_cast<float*>(_shared);
  }

  float* xpi=xpr+x.n1*x.n3;
  float* ypr=xpr+((2*x.n1*x.n3-1)/32+1)*32;
  float* ypi=ypr+y.n1*y.n3;

  int xs1=x.n3;
  int ys1=y.n3;
  int rs1=r.s1;
  int ytot=(y.n2-1)*y.n3+y.last;


  for(int i=0; i<x.n2; i++){
    int xn; if(i<x.n2-1) xn=x.n3; else xn=x.last; 
    loadg_tile(xpr,x,b,i,xn);

    for(int j=0; j<y.n2; j++){
      int yn; if(j<y.n2-1) yn=y.n3; else yn=y.last;
      loadg_tile(ypr,y,b,j,yn);

      __syncthreads();

      if(t<xn){
	float* _xpr=xpr+t;
	float* _xpi=xpi+t;
    
	for(int m1=-l1; m1<=l1; m1++){
	  int lower=-l-m1; if(lower<-l2) lower=-l2;
	  int upper=l-m1; if(upper>l2) upper=l2;
	  float x_r=0;
	  float x_i=0;

	  for(int ycol=0; ycol<yn; ycol++){

	    float* _ypr=ypr+ycol;
	    float* _ypi=ypi+ycol;
	    float* _rpr=r.arr+r.s0*b+r.s2*((i*x.n3+t)*ytot+(j*y.n3+ycol));
	    float* _rpi=r.arrc+r.s0*b+r.s2*((i*x.n3+t)*ytot+(j*y.n3+ycol));

	    for(int m2=lower; m2<=upper; m2++){
	      float c=cptr[(m1+l1)*L2+m2+l2];
	      const float y_r=_ypr[ys1*(m2+l2)];
	      const float y_i=_ypi[ys1*(m2+l2)];
	      const float g_r=_rpr[rs1*(m1+m2+l)];
	      const float g_i=_rpi[rs1*(m1+m2+l)];
	      x_r+=c*(g_r*y_r+g_i*y_i);
	      x_i+=c*(-g_r*y_i+g_i*y_r);
	    }
	  }

	  _xpr[xs1*(m1+l1)]+=x_r; 
	  _xpi[xs1*(m1+l1)]+=x_i;
	}

      }// end t<xn loop
      __syncthreads();

    }// end j<y.n2 loop

    saveg_tile(xpr,x,b,i,xn);
  }// end i<x.n2 loop

}


namespace GElib{


  void SO3partB_addCGproduct_back0_cu(const cnine::Ctensor3_view& x, cnine::Ctensor3_view r, const cnine::Ctensor3_view& y, 
    const int offs, const cudaStream_t& stream){

    const int xl=(x.n1-1)/2;
    const int yl=(y.n1-1)/2;
    const int l=(r.n1-1)/2;
    const int b=r.n0;

    r.arr+=r.s2*offs;
    r.arrc+=r.s2*offs;
    r.n2=x.n2*y.n2;

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    int clines=cnine::roundup(x.n1*y.n1,32)/32;

    // set tile sizes
    const int xn=std::min(x.n2,32);
    const int yn=std::min(y.n2,32);
    cnine::Ctensor4_view_t3 xtiled(x,xn);
    cnine::Ctensor4_view_t3 ytiled(y,yn);

    int nlines=cnine::roundup(xtiled.n1*xn*2,32)/32+
      cnine::roundup(ytiled.n1*yn*2,32)/32;

    if(nlines<=384){
      bool preloadCG=(nlines+clines<=384);
      SO3partB_addCGproduct_back0_tiled_kernel<<<b,cnine::roundup(xn,32),(nlines+preloadCG*clines)*128,stream>>>
	(xtiled,r,ytiled,Cptr,preloadCG);
      return;
    }

    cout<<"error"<<endl;

    /*
    if(nlines<=384){
      SO3partB_addCGproduct_back0_kernel<<<b,cnine::roundup(xg.n2*y.n2,32),nlines*128,stream>>>
	(xg,rg,y,Cptr);
    }else{
      cout<<"error"<<endl;
    }
    */
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
/*
__device__ int loadg1(const cnine::Ctensor3_view& x, float* dest, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J;
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
*/
