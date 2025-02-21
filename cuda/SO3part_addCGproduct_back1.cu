/*
 * This file is part of GElib, a C++/CUDA library for group equivariant 
 * tensor operations. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in orginal
 * or modified form) must retain this copyright notice and must be 
 * accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _SO3part_addCGproduct_back1_cu
#define _SO3part_addCGproduct_back1_cu

#include <cuda.h>
#include <cuda_runtime.h>

#include "SO3_CGbank.hpp"
#include "Ctensor5_view.hpp"
#include "utils.hpp"
#include "utils.cu"


extern GElib::SO3_CGbank SO3_cgbank;


__global__ void SO3part_addCGproduct_back1_tiled_kernel(const cnine::Ctensor5_view x, 
  const cnine::Ctensor4_view r, const cnine::Ctensor5_view y, int xremainder, int yremainder, 
  const int Cptr, float* cptr_global, const bool preloadCG){

  extern __shared__ unsigned char _shared[]; 
  const int b0=blockIdx.x;
  const int b1=blockIdx.y;
  const int t=threadIdx.x;

  int l1=(x.n2-1)/2;
  int l2=(y.n2-1)/2;
  int l=(r.n2-1)/2;
  int L2=y.n2;

  float* cptr;
  float* xpr;
  if(preloadCG){
    cptr=reinterpret_cast<float*>(_shared);
    xpr=cptr+((x.n2*y.n2-1)/32+1)*32;
    if(Cptr>=0) loadf(cptr,reinterpret_cast<float*>(cg_cmem)+Cptr,x.n2*y.n2);
    else loadf(cptr,cptr_global,x.n2*y.n2);
  }else{
    if(Cptr>=0) cptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
    else cptr=cptr_global;
    xpr=reinterpret_cast<float*>(_shared);
  }

  float* xpi=xpr+x.n2*x.n4;
  float* ypr=xpr+((2*x.n2*x.n4-1)/32+1)*32;
  float* ypi=ypr+y.n2*y.n4;

  int xs=x.s2;
  int ys=y.s2;
  int rs=r.s2;
  int ytot=y.n3*y.n4+yremainder;

  for(int j=0; j<=y.n3; j++){
    int yn=y.n4; 
    if(j==y.n3) yn=yremainder;
    if(yn==0) break;
    loadg_tile(ypr,y,j,yn);

    for(int i=0; i<=x.n3; i++){
      int xn=x.n4; 
      if(i==x.n3) xn=xremainder;
      if(xn==0) break;
      loadg_tile(xpr,x,i,xn);

      __syncthreads();

     if(t<yn){
	float* _ypr=ypr+t;
	float* _ypi=ypi+t;
    
	for(int m1=-l1; m1<=l1; m1++){
	  int lower=-l-m1; if(lower<-l2) lower=-l2;
	  int upper=l-m1; if(upper>l2) upper=l2;
	  float y_r=0;
	  float y_i=0;

	  for(int xcol=0; xcol<yn; xcol++){

	    float* _xpr=xpr+xcol;
	    float* _xpi=xpi+xcol;
	    float* _rpr=r.arr+r.s0*b0+r.s1*b1+r.s3*((i*x.n4+t)*ytot+(j*y.n4+ycol));
	    float* _rpi=r.arrc+r.s0*b0+r.s1*b1+r.s3*((i*x.n4+t)*ytot+(j*y.n4+ycol));

	    for(int m2=lower; m2<=upper; m2++){
	      float c=cptr[(m1+l1)*L2+m2+l2];
	      const float x_r=_xpr[xs*(m1+l1)];
	      const float x_i=_xpi[xs*(m1+l1)];
	      const float g_r=_rpr[rs*(m1+m2+l)];
	      const float g_i=_rpi[rs*(m1+m2+l)];
	      y_r+=c*(g_r*x_r+g_i*x_i);
	      y_i+=c*(-g_r*x_i+g_i*x_r);
	    }
	  }

	  _ypr[ys1*(m2+l2)]+=y_r; 
	  _ypi[ys1*(m2+l2)]+=y_i;
	}
     }

    }// for i

    saveg_tile(ypr,y,j,yn);

  }// for j

}


// --------------------------------------------------------------------------------------------------------------------


namespace GElib{


  void SO3part_addCGproduct_back1_cu(SO3part y, SO3part r, SO3part x, const int offs, const cudaStream_t& stream){

    GELIB_ASSRT(r.get_dev()==1);
    GELIB_ASSRT(x.get_dev()==1);
    GELIB_ASSRT(y.get_dev()==1);

    const int l1=x.getl();
    const int l2=y.getl();
    const int l=r.getl();
    const int L1=2*l1+1;
    const int L2=2*l2+1;
    GELIB_ASSRT(l>=std::abs(l1-l2) && l<=l1+l2);
    GELIB_ASSRT(r.getn()>=x.getn()*y.getn()+offs);

    r.canonicalize_to_4d();
    x.canonicalize_to_4d();
    y.canonicalize_to_4d();

    const int b=y.getb();
    r.promote_batch_to(b);
    x.promote_batch_to(b);

    const int g=y.getg();
    r.promote_grid_to(g);
    x.promote_grid_to(g);

    int xn=x.getn();
    int yn=cnine::roundup(y.getn(),32)*32;
    int xremainder=tile_channels(x,xn);
    int yremainder=tile_channels(y,yn);

    auto rv=view4_of(r);
    auto xv=view5_of(x);
    auto yv=view5_of(y);

    rv.arr+=rv.s3*offs;
    rv.arrc+=rv.s3*offs;
    //r.n2=x.n2*y.n2;

    float* cptr=nullptr;
    int Cptr=-1; //SO3_cgbank.getfC(xl,yl,l)/4; // const memory switched off for now
    if(Cptr<0) cptr=SO3_CGbank.get<float>(l1,l2,l,r.dev).get_arr();
    int clines=cnine::roundup(L1*L2,32)/32;

    int nlines=cnine::roundup(L1*xn*2,32)/32+
      cnine::roundup(L2*yn*2,32)/32;

    if(nlines<=384){
      bool preloadCG=(nlines+clines<=384);
      dim3 blocks(b,g);
      SO3part_addCGproduct_back1_tiled_kernel<<<blocks,cnine::roundup(yn,32),(nlines+preloadCG*clines)*128,stream>>>
	(yv,rv,xv,xremainder,yremainder,Cptr,cptr,preloadCG);
      return;
    }

    GELIB_ERROR("A single tile of the input and output tensors does not fit in shared memory.")

  }    


}


#endif 
