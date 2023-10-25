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

#ifndef _SO3partB_addCGproduct_cu
#define _SO3partB_addCGproduct_cu

#include <cuda.h>
#include <cuda_runtime.h>

#include "SO3_CGbank.hpp"
#include "GElibConfig.hpp"
#include "Ctensor3_view.hpp"
#include "Ctensor4_view.hpp"
#include "cuda_loaders.cu"


extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::GElibConfig* gelib_config;

#define maxl1_explicit 2
#define maxl_explicit 4

#include "SO3part_addCGproduct_subkernels.inc"


__global__ void SO3partB_addCGproduct_tiled_kernel(const cnine::Ctensor3_view r, const cnine::Ctensor4_view_t3 x, 
  const cnine::Ctensor4_view_t3 y, const int Cptr, float* cptr_global, const bool preloadCG){

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
    if(Cptr>=0) loadf(cptr,reinterpret_cast<float*>(cg_cmem)+Cptr,x.n1*y.n1);
    else loadf(cptr,cptr_global,x.n1*y.n1);
  }else{
    if(Cptr>=0) cptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
    else cptr=cptr_global;
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
      //int rn=xn*yn;
      loadg_tile(ypr,y,b,j,yn);

      __syncthreads();

      if(t<xn*yn){

	float* _xpr=xpr+t/yn;
	float* _xpi=xpi+t/yn;
    
	float* _ypr=ypr+t%yn;
	float* _ypi=ypi+t%yn;
    
	float* _rpr=r.arr+r.s0*b+r.s2*((i*x.n3+t/yn)*ytot+(j*y.n3+t%yn));
	float* _rpi=r.arrc+r.s0*b+r.s2*((i*x.n3+t/yn)*ytot+(j*y.n3+t%yn));

	for(int m=-l; m<=l; m++){
	  float r_r=0;
	  float r_i=0;
	  int lower=max(-l1,m-l2);
	  int upper=min(l1,m+l2);
	  for(int m1=lower; m1<=upper; m1++){
	    int m2=m-m1;
	    float c=cptr[(m1+l1)*L2+m2+l2];
	    const float x_r=_xpr[xs1*(m1+l1)];
	    const float x_i=_xpi[xs1*(m1+l1)];
	    const float y_r=_ypr[ys1*(m2+l2)];
	    const float y_i=_ypi[ys1*(m2+l2)];
	    r_r+=c*(x_r*y_r-x_i*y_i); 
	    r_i+=c*(x_r*y_i+x_i*y_r);
	  }
	  _rpr[rs1*(m+l)]+=r_r;
	  _rpi[rs1*(m+l)]+=r_i;
	}
      }
      __syncthreads();

    }
  }

}



typedef void (*CGPRODUCT_SUBKERNEL)(const float*, const float*, const float*, const float*, const int, float*, float*, int); 

template<CGPRODUCT_SUBKERNEL subkernel>
__global__ void SO3part_addCGproduct_explicit(const cnine::Ctensor3_view r, const cnine::Ctensor3_view x, 
  const cnine::Ctensor3_view y){

  extern __shared__ unsigned char _shared[]; 
  const int b=blockIdx.x;
  const int t=threadIdx.x;

  //int l1=(x.n1-1)/2;
  //int l2=(y.n1-1)/2;
  //int l=(r.n1-1)/2;
  //int L2=y.n1;
  //int L2=y.n1;

  float* xpr=reinterpret_cast<float*>(_shared);
  float* xpi=xpr+16; //xpr+x.n1;
  float* ypr=xpr+32; //xpr+((2*x.n1-1)/32+1)*32;
  float* ypi=ypr+y.n1*y.n2;
  loadg(y,ypr,b,t);

  for(int i=0; i<x.n2; i++){

    if(t<x.n1){
      xpr[t]=x.arr[b*x.s0+t*x.s1+i*x.s2];
      xpi[t]=x.arrc[b*x.s0+t*x.s1+i*x.s2];
    }

    if(t<y.n2){
      subkernel(xpr,xpi,ypr+t,ypi+t,y.n2,r.arr+b*r.s0+(i*y.n2+t)*r.s2,r.arrc+b*r.s0+(i*y.n2+t)*r.s2,r.s1);
    }
  }
  
}

namespace GElib{


  void SO3partB_addCGproduct_cu(cnine::Ctensor3_view r, const cnine::Ctensor3_view& x, const cnine::Ctensor3_view& y, 
    const int offs, const cudaStream_t& stream){

    const int xl=(x.n1-1)/2;
    const int yl=(y.n1-1)/2;
    const int l=(r.n1-1)/2;
    const int b=r.n0;

    r.arr+=r.s2*offs;
    r.arrc+=r.s2*offs;
    r.n2=x.n2*y.n2;
    //GELIB_CHECK(x.n2*y.n2<=1024,"Number of ouput channels can be at most 1024.")

    if(gelib_config && gelib_config->SO3part_CGkernels_explicit && xl<=maxl1_explicit && yl<=maxl1_explicit && l<=maxl_explicit){
      cout<<"Explicit!"<<endl;
      int nlines=1+cnine::roundup(2*y.n1*y.n2,32)/32;
      int l1=xl;
      int l2=yl;
      #include "SO3part_addCGproduct_explicit_calls.inc"
      return;
    }

    float* cptr=nullptr;
    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    if(Cptr<0) cptr=SO3_cgbank.getf(CGindex(xl,yl,l),r.dev).arrg;
    int clines=cnine::roundup(x.n1*y.n1,32)/32;

    // set tile sizes
    const int xn=std::min(x.n2,32);
    const int yn=std::min(y.n2,32);
    cnine::Ctensor4_view_t3 xtiled(x,xn);
    cnine::Ctensor4_view_t3 ytiled(y,yn);
    //cnine::Ctensor4_view_t3 rtiled(r,xn*yn);

    int nlines=cnine::roundup(xtiled.n1*xn*2,32)/32+
      cnine::roundup(ytiled.n1*yn*2,32)/32;

    if(nlines<=384){
      bool preloadCG=(nlines+clines<=384);
      //preloadCG=false;
      SO3partB_addCGproduct_tiled_kernel<<<b,cnine::roundup(xn*yn,32),(nlines+preloadCG*clines)*128,stream>>>
	(r,xtiled,ytiled,Cptr,cptr,preloadCG);
      return;
    }

    cout<<"error"<<endl;

  }    


}


#endif 



/*
__global__ void SO3partB_addCGproduct_kernel(const cnine::Ctensor3_view r, const cnine::Ctensor3_view x, 
  const cnine::Ctensor3_view y, const int Cptr, const bool preloadCG){

  extern __shared__ unsigned char _shared[]; 
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

  float* cptr;
  const float C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  if(preloadCG){
    cptr=rpr+((2*r.n1*rn-1)/32+1)*32;
    loadf(cptr,C_ptr,x.n1*y.n1,t);
  }else cptr=C_ptr;

  __syncthreads();

  if(t<rn){

    xpr=xpr+t/yn;
    xpi=xpi+t/yn;
    
    ypr=ypr+t%yn;
    ypi=ypi+t%yn;
    
    float* _rpr=rpr+t;
    float* _rpi=rpi+t;

    for(int m1=-l1; m1<=l1; m1++){
      const float x_r=xpr[xn*(m1+l1)];
      const float x_i=xpi[xn*(m1+l1)];
      int lower=-l-m1; if(lower<-l2) lower=-l2;
      int upper=l-m1; if(upper>l2) upper=l2;
      for(int m2=lower; m2<=upper; m2++){
	float c=C_ptr[(m1+l1)*L2+m2+l2];
	const float y_r=ypr[yn*(m2+l2)];
	const float y_i=ypi[yn*(m2+l2)];
	_rpr[rn*(m1+m2+l)]+=c*(x_r*y_r-x_i*y_i); 
	_rpi[rn*(m1+m2+l)]+=c*(x_r*y_i+x_i*y_r);
      }
    }
  }

  __syncthreads();
  saveg(r,rpr,b,t);
}
*/
