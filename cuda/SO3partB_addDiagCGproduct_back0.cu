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

#ifndef _SO3partB_addDiagCGproduct_back0_cu
#define _SO3partB_addDiagCGproduct_back0_cu

#include <cuda.h>
#include <cuda_runtime.h>

#include "SO3_CGbank.hpp"
#include "Ctensor3_view.hpp"
#include "cuda_loaders.cu"


extern GElib::SO3_CGbank SO3_cgbank;




// Process ncells number of cells in one call
__global__ void SO3partB_addDiagCGproduct_back0_kernel(const cnine::Ctensor3_view x, const cnine::Ctensor3_view r, 
  const cnine::Ctensor3_view y, const int Cptr, float* cptr_global, const bool preloadCG, const int ncells){

  extern __shared__ unsigned char _shared[]; 
  const int b=blockIdx.x;
  const int t=threadIdx.x;
  const int t0=t/x.n2; // cell selector
  const int t1=t%x.n2; // channel selector within cell
  const int actual_ncells=min(ncells,r.n0-b*ncells);

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

  float* xpi=xpr+actual_ncells*x.n1*x.n2;
  float* ypr=xpr+((2*actual_ncells*x.n1*x.n2-1)/32+1)*32;
  float* ypi=ypr+actual_ncells*y.n1*y.n2;

  int xs1=x.s1/2;
  int ys1=y.s1/2;
  int rs1=r.s1;

  loadf_strided(xpr,x.arr+b*ncells*x.s0,actual_ncells*x.n1*x.n2,2);
  loadf_strided(xpi,x.arrc+b*ncells*x.s0,actual_ncells*x.n1*x.n2,2);
  loadf_strided(ypr,y.arr+b*ncells*y.s0,actual_ncells*y.n1*y.n2,2);
  loadf_strided(ypi,y.arrc+b*ncells*y.s0,actual_ncells*y.n1*y.n2,2);
  __syncthreads();


  // this handles both the padding of the number of threads to a multiple of 32
  // and the padding of the number of blocks to a multiple of ncells  
  if(t0<actual_ncells){ 

    float* _xpr=xpr+t0*x.s0/2+t1;
    float* _xpi=xpi+t0*x.s0/2+t1;
    
    float* _ypr=ypr+t0*y.s0/2+t1;
    float* _ypi=ypi+t0*y.s0/2+t1;
    
    float* _rpr=r.arr+(b*ncells+t0)*r.s0+t1*r.s2;
    float* _rpi=r.arrc+(b*ncells+t0)*r.s0+t1*r.s2;

    for(int m1=-l1; m1<=l1; m1++){
      int lower=-l-m1; if(lower<-l2) lower=-l2;
      int upper=l-m1; if(upper>l2) upper=l2;
      float x_r=0;
      float x_i=0;

      for(int m2=lower; m2<=upper; m2++){
	float c=cptr[(m1+l1)*L2+m2+l2];
	const float y_r=_ypr[ys1*(m2+l2)];
	const float y_i=_ypi[ys1*(m2+l2)];
	const float g_r=_rpr[rs1*(m1+m2+l)];
	const float g_i=_rpi[rs1*(m1+m2+l)];
	x_r+=c*(g_r*y_r+g_i*y_i);
	x_i+=c*(-g_r*y_i+g_i*y_r);
      }

      _xpr[xs1*(m1+l1)]+=x_r; 
      _xpi[xs1*(m1+l1)]+=x_i;
    }
  }

  __syncthreads();
  savef_strided(xpr,x.arr+b*ncells*x.s0,actual_ncells*x.n1*x.n2,2);
  savef_strided(xpi,x.arrc+b*ncells*x.s0,actual_ncells*x.n1*x.n2,2);

}


__global__ void SO3partB_addDiagCGproduct_back0_tiled_kernel(const cnine::Ctensor4_view_t3 x, const cnine::Ctensor3_view r, 
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
  assert(x.n2==y.n2);

  for(int i=0; i<x.n2; i++){
    int xn; if(i<x.n2-1) xn=x.n3; else xn=x.last; 
    loadg_tile(xpr,x,b,i,xn);
    loadg_tile(ypr,y,b,i,xn);

    __syncthreads();

    if(t<xn){
      float* _xpr=xpr+t;
      float* _xpi=xpi+t;
	
      for(int m1=-l1; m1<=l1; m1++){
	int lower=-l-m1; if(lower<-l2) lower=-l2;
	int upper=l-m1; if(upper>l2) upper=l2;
	float x_r=0;
	float x_i=0;

	float* _ypr=ypr+t;
	float* _ypi=ypi+t;
	float* _rpr=r.arr+r.s0*b+r.s2*(i*x.n3+t);
	float* _rpi=r.arrc+r.s0*b+r.s2*(i*x.n3+t);

	for(int m2=lower; m2<=upper; m2++){
	  float c=cptr[(m1+l1)*L2+m2+l2];
	  const float y_r=_ypr[ys1*(m2+l2)];
	  const float y_i=_ypi[ys1*(m2+l2)];
	  const float g_r=_rpr[rs1*(m1+m2+l)];
	  const float g_i=_rpi[rs1*(m1+m2+l)];
	  x_r+=c*(g_r*y_r+g_i*y_i);
	  x_i+=c*(-g_r*y_i+g_i*y_r);
	}

	_xpr[xs1*(m1+l1)]+=x_r; 
	_xpi[xs1*(m1+l1)]+=x_i;
      }

    }// end t<xn loop
    __syncthreads();

    
    saveg_tile(xpr,x,b,i,xn);
  }// end i<x.n2 loop

}


namespace GElib{


  void SO3partB_addDiagCGproduct_back0_cu(const cnine::Ctensor3_view& x, cnine::Ctensor3_view r, const cnine::Ctensor3_view& y, 
    const int offs, const cudaStream_t& stream){

    GELIB_ASSERT(r.n0==x.n0,"Batch dimension mismatch.");
    GELIB_ASSERT(r.n0==y.n0,"Batch dimension mismatch.");
    GELIB_ASSERT(x.n2==y.n2,"Diag mismatch.");

    const int xl=(x.n1-1)/2;
    const int yl=(y.n1-1)/2;
    const int l=(r.n1-1)/2;
    const int b=r.n0;

    r.arr+=r.s2*offs;
    r.arrc+=r.s2*offs;
    r.n2=x.n2;

    float* cptr=nullptr;
    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    if(Cptr<0) cptr=SO3_cgbank.getf(CGindex(xl,yl,l),r.dev).arrg;
    int clines=cnine::roundup(x.n1*y.n1,32)/32;


    // If the number of channels is 32 or less, try and process multiple cells in one thread-block
    if(x.n2<=32){
      int ncells=32/x.n2;
      ncells=std::min(ncells,384*32/(2*x.n1*x.n2+2*y.n1*y.n2));
      int nlines=cnine::roundup(2*ncells*x.n1*x.n2,32)/32+
	cnine::roundup(2*ncells*y.n1*y.n2,32)/32;
      if(ncells>0 && nlines<=384){
	bool preloadCG=(nlines+clines<=384);
	//cout<<"Launching addDiagCGproduct_kernel_back0 with ncells="<<ncells<<" nblocks="<<cnine::roundup(b,ncells)/ncells<<" and nthreads="<<cnine::roundup(ncells*x.n2,32)<<endl; 
	SO3partB_addDiagCGproduct_back0_kernel<<<cnine::roundup(b,ncells)/ncells,cnine::roundup(ncells*x.n2,32),
	  (nlines+preloadCG*clines)*128,stream>>>
	  (x,r,y,Cptr,cptr,preloadCG,ncells);
	return;
      }
    }

    // Otherwise tile the inputs to chunks of width 32
    const int tilesize=std::min(x.n2,32);
    cnine::Ctensor4_view_t3 xtiled(x,tilesize);
    cnine::Ctensor4_view_t3 ytiled(y,tilesize);
    int nlines=cnine::roundup(xtiled.n1*tilesize*2,32)/32+
      cnine::roundup(ytiled.n1*tilesize*2,32)/32;
    if(nlines<=384){
      bool preloadCG=(nlines+clines<=384);
      SO3partB_addDiagCGproduct_back0_tiled_kernel<<<b,cnine::roundup(tilesize,32),(nlines+preloadCG*clines)*128,stream>>>
	(xtiled,r,ytiled,Cptr,cptr,preloadCG);
      return;
    }

    GELIB_ERROR("Inputs too large to load in shared memory.");

  }    


}


#endif 



