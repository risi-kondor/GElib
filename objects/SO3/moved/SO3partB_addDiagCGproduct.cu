/*
 * This file is part of the GElib-cuda library
 * 
 * Copyright (c) 2022, Imre Risi Kondor
 *
 * All rights reserved. Copying, distributing, modifying or using 
 * this file without the copyright holder's permission is prohibited. 
*/ 

#ifndef _SO3partB_addDiagCGproduct_cu
#define _SO3partB_addDiagCGproduct_cu

#include <cuda.h>
#include <cuda_runtime.h>

#include "SO3_CGbank.hpp"
#include "Ctensor3_view.hpp"
#include "Ctensor4_view.hpp"
#include "cuda_loaders.cu"


extern GElib::SO3_CGbank SO3_cgbank;


__global__ void SO3partB_addDiagCGproduct_tiled_kernel(const cnine::Ctensor3_view r, const cnine::Ctensor4_view_t3 x, 
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


namespace GElib{


  void SO3partB_addDiagCGproduct_cu(cnine::Ctensor3_view r, const cnine::Ctensor3_view& x, const cnine::Ctensor3_view& y, 
    const int offs, const cudaStream_t& stream){

    const int xl=(x.n1-1)/2;
    const int yl=(y.n1-1)/2;
    const int l=(r.n1-1)/2;
    const int b=r.n0;

    r.arr+=r.s2*offs;
    r.arrc+=r.s2*offs;
    r.n2=x.n2;
    GELIB_CHECK(x.n2==y.n2,"Diag mismatch.");
    //GELIB_CHECK(x.n2*y.n2<=1024,"Number of ouput channels can be at most 1024.")

    float* cptr=nullptr;
    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    if(Cptr<0) cptr=SO3_cgbank.getf(CGindex(xl,yl,l),r.dev).arrg;
    int clines=cnine::roundup(x.n1*y.n1,32)/32;

    const int tilesize=std::min(x.n2,32);
    cnine::Ctensor4_view_t3 xtiled(x,tilesize);
    cnine::Ctensor4_view_t3 ytiled(y,tilesize);

    int nlines=cnine::roundup(xtiled.n1*tilesize*2,32)/32+
      cnine::roundup(ytiled.n1*tilesize*2,32)/32;

    if(nlines<=384){
      bool preloadCG=(nlines+clines<=384);
      //preloadCG=false;
      SO3partB_addDiagCGproduct_tiled_kernel<<<b,cnine::roundup(tilesize,32),(nlines+preloadCG*clines)*128,stream>>>
	(r,xtiled,ytiled,Cptr,cptr,preloadCG);
      return;
    }

    cout<<"error"<<endl;

  }    


}


#endif 



