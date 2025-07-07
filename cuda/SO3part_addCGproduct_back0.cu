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

#ifndef _SO3part_addCGproduct_back0_cu
#define _SO3part_addCGproduct_back0_cu

#include <cuda.h>
#include <cuda_runtime.h>

#include "SO3CGbank.hpp"
#include "GPUtensor.hpp"
#include "SO3part.hpp"
#include "utils.hpp"
#include "utils.cu"


extern GElib::SO3CGbank SO3_CGbank;
//extern __device__ __constant__ unsigned char cg_cmem[]; 


__global__ void SO3part_addCGproduct_back0_kernel(const cnine::GPUtensor<float,5> x, 
  const cnine::GPUtensor<float,5> r, const cnine::GPUtensor<float,5> y,  
  const int Cptr, float* cptr_global, const bool preloadCG){

  extern __shared__ unsigned char _shared[]; 
  const int b0=blockIdx.x;
  const int b1=blockIdx.y;
  const int b2=blockIdx.z;
  const int t=threadIdx.x;

  int l1=(x.dims[3]-1)/2;
  int l2=(y.dims[3]-1)/2;
  int l=(r.dims[3]-1)/2;
  int L1=x.dims[3];
  int L2=y.dims[3];
  int xn=x.dims[4];
  int yn=y.dims[4];

  float* cptr;
  float* xpr;
  if(preloadCG){
    cptr=reinterpret_cast<float*>(_shared);
    xpr=cptr+((L1*L2-1)/32+1)*32;
    loadf(cptr,cptr_global,L1*L2);
  }else{
    cptr=cptr_global;
    xpr=reinterpret_cast<float*>(_shared);
  }

  float* xpi=xpr+L1*xn;
  float* ypr=xpr+((2*L1*xn-1)/32+1)*32;
  float* ypi=ypr+L2*yn;
  float* rpr=r.arr+r.strides[0]*b0+r.strides[1]*b1+r.strides[2]*b2+t*yn*r.strides[4];

  loadf(xpr,x.arr+x.strides[0]*b0+x.strides[1]*b1+x.strides[2]*b2,L1*xn,x.strides[4]);
  loadf(xpi,x.arr+x.strides[0]*b0+x.strides[1]*b1+x.strides[2]*b2+1,L1*xn,x.strides[4]);

  loadf(ypr,y.arr+y.strides[0]*b0+y.strides[1]*b1+y.strides[2]*b2,L2*yn,y.strides[4]);
  loadf(ypi,y.arr+y.strides[0]*b0+y.strides[1]*b1+y.strides[2]*b2+1,L2*yn,y.strides[4]);

  __syncthreads();

  int xs=xn;
  int ys=yn;
  int rs=r.strides[3];

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
	float* _rpr=rpr+ycol*r.strides[4];
	float* _rpi=_rpr+1;

	for(int m2=lower; m2<=upper; m2++){
	  float c=cptr[(m1+l1)*L2+m2+l2];
	  const float y_r=_ypr[ys*(m2+l2)];
	  const float y_i=_ypi[ys*(m2+l2)];
	  const float g_r=_rpr[rs*(m1+m2+l)];
	  const float g_i=_rpi[rs*(m1+m2+l)];
	  x_r+=c*(g_r*y_r+g_i*y_i);
	  x_i+=c*(-g_r*y_i+g_i*y_r);
	}
      }
      _xpr[xs*(m1+l1)]+=x_r; 
      _xpi[xs*(m1+l1)]+=x_i;
    }
  }

  savef(x.arr+x.strides[0]*b0+x.strides[1]*b1+x.strides[2]*b2,xpr,L1*xn,x.strides[4]);
  savef(x.arr+x.strides[0]*b0+x.strides[1]*b1+x.strides[2]*b2+1,xpi,L1*xn,x.strides[4]);

}


__global__ void SO3part_addCGproduct_back0_tiled_kernel(const cnine::GPUtensor<float,5> x, 
  const cnine::GPUtensor<float,5> r, const cnine::GPUtensor<float,5> y,  
  int xn, int yn, const int Cptr, float* cptr_global, const bool preloadCG){

  extern __shared__ unsigned char _shared[]; 
  const int b0=blockIdx.x;
  const int b1=blockIdx.y;
  const int b2=blockIdx.z;
  const int t=threadIdx.x;

  int l1=(x.dims[3]-1)/2;
  int l2=(y.dims[3]-1)/2;
  int l=(r.dims[3]-1)/2;
  int L1=x.dims[3];
  int L2=y.dims[3];
  int xN=x.dims[4]/xn;
  int yN=y.dims[4]/yn;

  float* cptr;
  float* xpr;
  if(preloadCG){
    cptr=reinterpret_cast<float*>(_shared);
    xpr=cptr+((L1*L2-1)/32+1)*32;
    loadf(cptr,cptr_global,L1*L2);
  }else{
    cptr=cptr_global;
    xpr=reinterpret_cast<float*>(_shared);
  }

  float* xpi=xpr+L1*xn;
  float* ypr=xpr+((2*L1*xn-1)/32+1)*32;
  float* ypi=ypr+L2*yn;

  float* rarr=r.arr+r.strides[0]*b0+r.strides[1]*b1+r.strides[2]*b2;
  float* xarr=x.arr+x.strides[0]*b0+x.strides[1]*b1+x.strides[2]*b2;
  float* yarr=y.arr+y.strides[0]*b0+y.strides[1]*b1+y.strides[2]*b2;

  int xs=xn;
  int ys=yn;
  int rs=r.strides[3];

  for(int i=0; i<=xN; i++){
    int _xn=xn;
    if(i==xN) _xn=x.dims[4]%xn;
    if(_xn==0) break;

    load_tile(xpr,xarr+i*xn*x.strides[4],L1,_xn,x.strides[3],x.strides[4]);
    load_tile(xpi,xarr+i*xn*x.strides[4]+1,L1,_xn,x.strides[3],x.strides[4]);
    __syncthreads();

    for(int j=0; j<=yN; j++){
      int _yn=yn;
      if(j==yN) _yn=y.dims[4]%yn;
      if(_yn==0) break;

      load_tile(ypr,yarr+j*yn*y.strides[4],L2,_yn,y.strides[3],y.strides[4]);
      load_tile(ypi,yarr+j*yn*y.strides[4]+1,L2,_yn,y.strides[3],y.strides[4]);
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
	    float* _rpr=rarr+((i*xn+t)*y.dims[4]+j*yn+ycol)*r.strides[4];
	    float* _rpi=_rpr+1;

	    for(int m2=lower; m2<=upper; m2++){
	      float c=cptr[(m1+l1)*L2+m2+l2];
	      const float y_r=_ypr[ys*(m2+l2)];
	      const float y_i=_ypi[ys*(m2+l2)];
	      const float g_r=_rpr[rs*(m1+m2+l)];
	      const float g_i=_rpi[rs*(m1+m2+l)];
	      x_r+=c*(g_r*y_r+g_i*y_i);
	      x_i+=c*(-g_r*y_i+g_i*y_r);
	    }
	  }
	  _xpr[xs*(m1+l1)]+=x_r; 
	  _xpi[xs*(m1+l1)]+=x_i;
	}
      }

    }// for j

    save_tile(xarr+i*xn*x.strides[4],xpr,L1,_xn,x.strides[3],x.strides[4]);
    save_tile(xarr+i*xn*x.strides[4]+1,xpi,L1,_xn,x.strides[3],x.strides[4]);
    __syncthreads();

  }// for i

}


// --------------------------------------------------------------------------------------------------------------------


namespace GElib{


  void SO3part_addCGproduct_back0_cu(const SO3part<float>& x, const SO3part<float>& r, const SO3part<float>& y, const int offs, const cudaStream_t& stream){

    if(r.get_dev()!=1) GELIB_SKIP("SO3part r must be on GPU");
    if(x.get_dev()!=1) GELIB_SKIP("SO3part x must be on GPU");
    if(y.get_dev()!=1) GELIB_SKIP("SO3part y must be on GPU");

    if(r.ndims()!=5) GELIB_SKIP("SO3part r must be 5D");
    if(x.ndims()!=5) GELIB_SKIP("SO3part x must be 5D");
    if(y.ndims()!=5) GELIB_SKIP("SO3part y must be 5D");

    auto rsubdims=r.dims.chunk(0,3);
    if(x.dims.chunk(0,3)!=rsubdims) GELIB_SKIP("leading dimensions of x and r must be same");
    if(y.dims.chunk(0,3)!=rsubdims) GELIB_SKIP("leading dimensions of y and r must be same");
    if(r.dims[0]>65535) GELIB_SKIP("dims[0] exceeds 65535");
    if(r.dims[1]>65535) GELIB_SKIP("dims[1] exceeds 65535");
    if(r.dims[2]>65535) GELIB_SKIP("dims[2] exceeds 65535");
    if((size_t)(r.dims[0])*r.dims[1]*r.dims[2]>INT_MAX) GELIB_SKIP("product of block dimensions exceeds 2^31-1");

    const int l1=x.getl();
    const int l2=y.getl();
    const int l=r.getl();
    const int L1=2*l1+1;
    const int L2=2*l2+1;
    if(l<std::abs(l1-l2) || l>l1+l2) GELIB_SKIP("|l1-l_2| <= l <= l1+l2 not satisfied");

    int xn=x.dims[4];
    int yn=y.dims[4];
    if(xn*yn+offs>r.dims[4]) GELIB_SKIP("fragment dimension of r not large enough");

    cnine::GPUtensor<float,5> rv(r);
    cnine::GPUtensor<float,5> xv(x);
    cnine::GPUtensor<float,5> yv(y);
    rv.arr+=r.strides[4]*offs;

    float* cptr=SO3_CGbank.get<float>(l1,l2,l,r.dev).get_arr(); 
    int clines=cnine::roundup(L1*L2,32)/32;
    int nlines=cnine::roundup(L1*xn*2,32)/32+cnine::roundup(L2*yn*2,32)/32;

    bool tiled=xn>1024;
    if(x.strides[3]!=x.strides[4]*x.dims[4]) tiled=true;
    if(y.strides[3]!=y.strides[4]*y.dims[4]) tiled=true;
    if(nlines+clines>384) tiled=true;

    if(!tiled){
      bool preloadCG=(nlines+clines<=384);
      dim3 blocks(r.dims[0],r.dims[1],r.dims[1]);
      SO3part_addCGproduct_back0_kernel<<<blocks,cnine::roundup(xn,32),(nlines+preloadCG*clines)*128,stream>>>
	(xv,rv,yv,-1,cptr,true);
      return;
    }

    if(tiled){
      xn=std::min(xn,1024);
      int nlines=cnine::roundup(L1*xn*2,32)/32+cnine::roundup(L2*yn*2,32)/32;
      int conservative_clines=clines;
      if(conservative_clines>200) conservative_clines=0; // try preloading CG if it uses at most 200 lines

      if(nlines<=384){ // should always be true
	bool preloadCG=(nlines+clines<=384);
	dim3 blocks(r.dims[0],r.dims[1],r.dims[1]);
	SO3part_addCGproduct_back0_tiled_kernel<<<blocks,cnine::roundup(xn,32),(nlines+preloadCG*clines)*128,stream>>>
	  (xv,rv,yv,xn,yn,-1,cptr,true);
	return;
      }

    }


    GELIB_SKIP("tiled kernel not supported.");

  }    


}


#endif 
