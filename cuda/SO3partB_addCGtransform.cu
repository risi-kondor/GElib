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
#include "Ctensor3_view.hpp"
#include "Ctensor4_view.hpp"
#include "cuda_loaders.cu"


extern GElib::SO3_CGbank SO3_cgbank;


__global__ void SO3partB_addCGtransform_kernel(const cnine::Ctensor3_view r, const cnine::Ctensor4_view x, 
  const int Cptr, float* cptr_global, const bool preloadCG){

  extern __shared__ unsigned char _shared[]; 
  const int b=blockIdx.x;
  const int t=threadIdx.x;

  int l1=(x.n1-1)/2;
  int l2=(x.n2-1)/2;
  int l=(r.n1-1)/2;
  //int L2=y.n1;

  float* cptr;
  float* xpr;
  if(preloadCG){
    cptr=reinterpret_cast<float*>(_shared);
    xpr=cptr+((x.n1*x.n2-1)/32+1)*32;
    if(Cptr>=0) loadf(cptr,reinterpret_cast<float*>(cg_cmem)+Cptr,x.n1*x.n2);
    else loadf(cptr,cptr_global,x.n1*x.n2);
  }else{
    if(Cptr>=0) cptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
    else cptr=cptr_global;
    xpr=reinterpret_cast<float*>(_shared);
  }

  

}


namespace GElib{


  void SO3partB_addCGtransform_cu(cnine::Ctensor3_view r, const cnine::Ctensor4_view& x,  
    const int offs, const cudaStream_t& stream){

    const int xl=(x.n1-1)/2;
    const int yl=(x.n2-1)/2;
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
