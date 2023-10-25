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

#ifndef _SO3part_addCGtransform_cu
#define _SO3part_addCGtransform_cu

#include <cuda.h>
#include <cuda_runtime.h>

#include "SO3_CGbank.hpp"
#include "Ctensor3_view.hpp"
#include "Ctensor4_view.hpp"
#include "cuda_loaders.cu"


extern GElib::SO3_CGbank SO3_cgbank;
//extern long int opcount;

// Process ncells number of cells in one call
__global__ void SO3part_addCGtransform_kernel(const cnine::Ctensor3_view r, const cnine::Ctensor4_view x, 
  const int Cptr, float* cptr_global, const bool preloadCG, const int ncells){

  extern __shared__ unsigned char _shared[]; 
  const int b=blockIdx.x;
  const int t=threadIdx.x;
  const int t0=t/x.n3; // cell selector
  const int t1=t%x.n3; // channel selector within cell
  const int actual_ncells=min(ncells,r.n0-b*ncells);

  int l1=(x.n1-1)/2;
  int l2=(x.n2-1)/2;
  int l=(r.n1-1)/2;
  int L2=x.n2;

  float* cptr;
  //float* xpr;
  if(preloadCG){
    cptr=reinterpret_cast<float*>(_shared);
    //xpr=cptr+((x.n1*x.n2-1)/32+1)*32;
    if(Cptr>=0) loadf(cptr,reinterpret_cast<float*>(cg_cmem)+Cptr,x.n1*x.n2);
    else loadf(cptr,cptr_global,x.n1*x.n2);
  }else{
    if(Cptr>=0) cptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
    else cptr=cptr_global;
    //xpr=reinterpret_cast<float*>(_shared);
  }

  //loadf(xpr,x.arr+b*ncells*x.s0,actual_ncells*x.n1*x.n2*x.n3);
  __syncthreads();

  if(t0<actual_ncells){ 

    int xs1=x.s1;
    int xs2=x.s2;
    int rs1=r.s1;
    
    //float* _xpr=xpr+t0*x.s0+t1;
    float* _xpr=x.arr+b*ncells*x.s0+t0*x.s0+t1*x.s3;
    float* _rpr=r.arr+(b*ncells+t0)*r.s0+t1*r.s2;

    for(int m=-l; m<=l; m++){
      float r_r=0;
      int lower=max(-l1,m-l2);
      int upper=min(l1,m+l2);
      for(int m1=lower; m1<=upper; m1++){
	int m2=m-m1;
	float c=cptr[(m1+l1)*L2+m2+l2];
	r_r+=c*_xpr[xs1*(m1+l1)+xs2*(m2+l2)];
      }
      _rpr[rs1*(m+l)]+=r_r;
    }
    
  }

  __syncthreads();

}

/*
__global__ void SO3part_addCGtransform_tiled_kernel(const cnine::Ctensor3_view r, const cnine::Ctensor4_view_t3 x, 
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
    
      float* _ypr=ypr+t;
      float* _ypi=ypi+t;
    
      float* _rpr=r.arr+r.s0*b+r.s2*(i*x.n3+t);
      float* _rpi=r.arrc+r.s0*b+r.s2*(i*x.n3+t);

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
*/

namespace GElib{


  void SO3part_addCGtransform_cu(cnine::Ctensor3_view r, cnine::Ctensor4_view x, 
    const int offs, const cudaStream_t& stream){

    GELIB_ASSERT(r.n0==x.n0,"Batch dimension mismatch.");

    const int b=r.n0;
    const int l1=(x.n1-1)/2;
    const int l2=(x.n2-1)/2;
    const int l=(r.n1-1)/2;
    const int n=2*x.n3;

    // convert to real tensors by doubling channels
    r.arr+=r.s2*offs;
    r.arrc+=r.s2*offs;
    r.n2=n;
    x.n3=n;
    r.s2=1;
    x.s3=1;

    float* cptr=nullptr;
    int Cptr=SO3_cgbank.getfC(l1,l2,l)/4;
    if(Cptr<0) cptr=SO3_cgbank.getf(CGindex(l1,l2,l),r.dev).arrg;
    int clines=cnine::roundup(x.n1*x.n2,32)/32;

    // If the number of channels is 32 or less, process multiple cells in one thread-block
    if(n<=32){
      int ncells=32/n;
      ncells=std::min(ncells,380*32/(x.n1*x.n2*x.n3));
      int nlines=0; //cnine::roundup(ncells*x.n1*x.n2*x.n3,32)/32;
      if(ncells>0 && nlines<=384){
	bool preloadCG=(nlines+clines<=384);
	SO3part_addCGtransform_kernel<<<cnine::roundup(b,ncells)/ncells,cnine::roundup(ncells*n,32),
	  (nlines+preloadCG*clines)*128,stream>>>
	  (r,x,Cptr,cptr,preloadCG,ncells);
	return;
      }
    }


    /*
    // Otherwise tile the inputs to chunks of width 32
    const int tilesize=std::min(x.n2,32);
    cnine::Ctensor4_view_t3 xtiled(x,tilesize);
    cnine::Ctensor4_view_t3 ytiled(y,tilesize);
    int nlines=cnine::roundup(xtiled.n1*tilesize*2,32)/32+
      cnine::roundup(ytiled.n1*tilesize*2,32)/32;
    
    if(nlines<=384){
      bool preloadCG=(nlines+clines<=384);
      SO3part_addCGtransform_tiled_kernel<<<b,cnine::roundup(tilesize,32),(nlines+preloadCG*clines)*128,stream>>>
	(r,xtiled,ytiled,Cptr,cptr,preloadCG);
      return;
    }
    */

    GELIB_ERROR("Inputs too large to load in shared memory.");
  }    


}


#endif 



