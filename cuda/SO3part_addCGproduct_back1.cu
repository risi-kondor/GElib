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

#include "SO3CGbank.hpp"
#include "SO3part.hpp"
#include "utils.hpp"
#include "utils.cu"


extern GElib::SO3CGbank SO3_CGbank;
//extern __device__ __constant__ unsigned char cg_cmem[]; 


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
    //if(Cptr>=0) loadf(cptr,reinterpret_cast<float*>(cg_cmem)+Cptr,x.n2*y.n2);
    //else 
    loadf(cptr,cptr_global,x.n2*y.n2);
  }else{
    //if(Cptr>=0) cptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
    //else 
    cptr=cptr_global;
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
    
	for(int m2=-l2; m2<=l2; m2++){
	  int lower=-l-m2; if(lower<-l1) lower=-l1;
	  int upper=l-m2; if(upper>l1) upper=l1;
	  float y_r=0;
	  float y_i=0;

	  for(int xcol=0; xcol<xn; xcol++){

	    float* _xpr=xpr+xcol;
	    float* _xpi=xpi+xcol;
	    float* _rpr=r.arr+r.s0*b0+r.s1*b1+r.s3*((i*x.n4+xcol)*ytot+(j*y.n4+t));
	    float* _rpi=r.arrc+r.s0*b0+r.s1*b1+r.s3*((i*x.n4+xcol)*ytot+(j*y.n4+t));

	    for(int m1=lower; m1<=upper; m1++){
	      float c=cptr[(m1+l1)*L2+m2+l2];
	      const float x_r=_xpr[xs*(m1+l1)];
	      const float x_i=_xpi[xs*(m1+l1)];
	      const float g_r=_rpr[rs*(m1+m2+l)];
	      const float g_i=_rpi[rs*(m1+m2+l)];
	      y_r+=c*(g_r*x_r+g_i*x_i);
	      y_i+=c*(-g_r*x_i+g_i*x_r);
	    }
	  }

	  _ypr[ys*(m2+l2)]+=y_r; 
	  _ypi[ys*(m2+l2)]+=y_i;
	}
     }

    }// for i

    saveg_tile(ypr,y,j,yn);

  }// for j

}


// --------------------------------------------------------------------------------------------------------------------


namespace GElib{


  void SO3part_addCGproduct_back1_cu(const SO3part<float>& y, const SO3part<float>& r, const SO3part<float>& x, const int offs, const cudaStream_t& stream){

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

    int xn=x.getn();
    int yn=cnine::roundup(y.getn(),32)*32;
    int xremainder=x.dims.back()%xn;
    int yremainder=y.dims.back()%yn;

    if(r.dims[2]==1){

      auto rv=view4_of(r.fuse(1,2));
      auto xv=tiled_view4_of(x.fuse(1,2),xn);
      auto yv=tiled_view4_of(y.fuse(1,2),yn);

      rv.arr+=rv.s3*offs;
      rv.arrc+=rv.s3*offs;
      //r.n2=x.n2*y.n2;

      float* cptr=SO3_CGbank.get<float>(l1,l2,l,r.dev).get_arr();
      int clines=cnine::roundup(L1*L2,32)/32;
      int nlines=cnine::roundup(L1*xn*2,32)/32+cnine::roundup(L2*yn*2,32)/32;
      
      if(nlines<=384){
	bool preloadCG=(nlines+clines<=384);
	dim3 blocks(r.dims[0],r.dims[1]);
	SO3part_addCGproduct_back1_tiled_kernel<<<blocks,cnine::roundup(yn,32),(nlines+preloadCG*clines)*128,stream>>>
	  (yv,rv,xv,xremainder,yremainder,-1,cptr,preloadCG);
	return;
      }
      
      GELIB_ERROR("A single tile of the input and output tensors does not fit in shared memory.");
    }

    GELIB_ERROR("5D SO3parts not supported.");
  }    


}


#endif 
