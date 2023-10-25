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

#ifndef _SO3partB_addCGsquare_cu
#define _SO3partB_addCGsquare_cu

#include <cuda.h>
#include <cuda_runtime.h>

#include "SO3_CGbank.hpp"
#include "Ctensor3_view.hpp"
#include "Ctensor4_view.hpp"
#include "cuda_loaders.cu"


extern GElib::SO3_CGbank SO3_cgbank;


__global__ void SO3partB_addCGsquare_tiled_kernel(const cnine::Ctensor3_view r, const cnine::Ctensor4_view_t3 x, 
  const int Cptr, float* cptr_global, const bool preloadCG){

  extern __shared__ unsigned char _shared[]; 
  const int b=blockIdx.x;
  const int t=threadIdx.x;

  int l1=(x.n1-1)/2;
  int l=(r.n1-1)/2;
  int L2=x.n1;

  float* cptr;
  float* xpr;
  if(preloadCG){
    cptr=reinterpret_cast<float*>(_shared);
    xpr=cptr+((x.n1*x.n1-1)/32+1)*32;
    if(Cptr>=0) loadf(cptr,reinterpret_cast<float*>(cg_cmem)+Cptr,x.n1*x.n1);
    else loadf(cptr,cptr_global,x.n1*x.n1);
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


namespace GElib{


  void SO3partB_addCGsquare_cu(cnine::Ctensor3_view r, const cnine::Ctensor3_view& x,  
    const int offs, const cudaStream_t& stream){

    const int xl=(x.n1-1)/2;
    const int l=(r.n1-1)/2;
    const int b=r.n0;
    const int diag=1-(2*xl-l)%2;

    r.arr+=r.s2*offs;
    r.arrc+=r.s2*offs;
    r.n2=x.n2*(x.n2-1)/2+x.n2*diag;
    //GELIB_CHECK(x.n2*y.n2<=1024,"Number of ouput channels can be at most 1024.")

    float* cptr=nullptr;
    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    if(Cptr<0) cptr=SO3_cgbank.getf(CGindex(xl,yl,l),r.dev).arrg;
    int clines=cnine::roundup(r.n2,32)/32;

    // set tile sizes
    const int xn=std::min(x.n2,32);
    cnine::Ctensor4_view_t3 xtiled(x,xn);
    //cnine::Ctensor4_view_t3 rtiled(r,xn*yn);

    int nlines=2*cnine::roundup(xtiled.n1*xn*2,32)/32;

    if(nlines<=384){
      bool preloadCG=(nlines+clines<=384);
      SO3partB_addCGsquare_tiled_kernel<<<b,cnine::roundup(xn*xn,32),(nlines+preloadCG*clines)*128,stream>>>
	(r,xtiled,Cptr,cptr,preloadCG);
      return;
    }

    cout<<"error"<<endl;

  }    


}


#endif 

