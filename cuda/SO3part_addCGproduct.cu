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

#ifndef _SO3part_addCGproduct_cu
#define _SO3part_addCGproduct_cu

#include <cuda.h>
#include <cuda_runtime.h>

#include "SO3CGbank.hpp"
#include "Ctensor5_view.hpp"
#include "GPUtensor.hpp"
#include "utils.hpp"
#include "utils.cu"
#include "SO3part.hpp"


extern GElib::SO3CGbank SO3_CGbank;
//extern __device__ __constant__ unsigned char cg_cmem[]; 


__device__ void SO3_CGproduct_kernel(float* _rpr, float* _rpi, int rs, 
  const float* _xpr, const float* _xpi, const int xs, 
  const float* _ypr, const float* _ypi, const int ys,
  const int l1, const int l2, const int l, const int L2, float* cptr){

  for(int m=-l; m<=l; m++){
    float r_r=0;
    float r_i=0;
    int lower=max(-l1,m-l2);
    int upper=min(l1,m+l2);
    for(int m1=lower; m1<=upper; m1++){
      int m2=m-m1;
      float c=cptr[(m1+l1)*L2+m2+l2];
      const float x_r=_xpr[xs*(m1+l1)];
      const float x_i=_xpi[xs*(m1+l1)];
      const float y_r=_ypr[ys*(m2+l2)];
      const float y_i=_ypi[ys*(m2+l2)];
      r_r+=c*(x_r*y_r-x_i*y_i); 
      r_i+=c*(x_r*y_i+x_i*y_r);
    }
    _rpr[rs*(m+l)]+=r_r;
    _rpi[rs*(m+l)]+=r_i;
  }
}



__global__ void SO3part_addCGproduct_kernel(const cnine::GPUtensor<float,5> r, 
  const cnine::GPUtensor<float,5> x, const cnine::GPUtensor<float,5> y,  
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
  int nx=x.dims[4];
  int ny=y.dims[4];

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

  float* xpi=xpr+L1*nx;
  float* ypr=xpr+((2*L1*nx-1)/32+1)*32;
  float* ypi=ypr+L2*ny;

  loadf(xpr,x.arr+x.strides[0]*b0+x.strides[1]*b1+x.strides[2]*b2,L1*nx,x.strides[4]);
  loadf(xpi,x.arr+x.strides[0]*b0+x.strides[1]*b1+x.strides[2]*b2+1,L1*nx,x.strides[4]);

  loadf(ypr,y.arr+y.strides[0]*b0+y.strides[1]*b1+y.strides[2]*b2,L2*ny,y.strides[4]);
  loadf(ypi,y.arr+y.strides[0]*b0+y.strides[1]*b1+y.strides[2]*b2+1,L2*ny,y.strides[4]);

  __syncthreads();

  if(t<nx*ny){
    int i=t/ny;
    int j=t%ny;
    float* rpr=r.arr+r.strides[0]*b0+r.strides[1]*b1+r.strides[2]*b2+r.strides[4]*t;
    SO3_CGproduct_kernel(rpr,rpr+1,r.strides[3],xpr+i,xpi+i,nx,ypr+j,ypi+j,ny,l1,l2,l,L2,cptr); 
  }

}


__global__ void SO3part_addCGproduct_tiled_kernel
(const cnine::GPUtensor<float,5> r, const cnine::GPUtensor<float,5> x, const cnine::GPUtensor<float,5> y,  
  const int xn, const int yn, const int Cptr, float* cptr_global, const bool preloadCG){

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
    
      if(t<_xn*_yn){
	int _i=t/_yn;
	int _j=t%_yn;
	float* _rarr=rarr+((i*xn+_i)*y.dims[4]+j*yn+_j)*r.strides[4];
	SO3_CGproduct_kernel(_rarr,_rarr+1,r.strides[3],xpr+_i,xpi+_i,_xn,ypr+_j,ypi+_j,_yn,l1,l2,l,L2,cptr); 
      }
      __syncthreads();

    }// for j

  }// for i

}


// --------------------------------------------------------------------------------------------------------------------


namespace GElib{


  void SO3part_addCGproduct_cu(const SO3part<float>& r, const SO3part<float>& x, const SO3part<float>& y, const int offs, const cudaStream_t& stream){

    if(r.get_dev()!=1) GELIB_SKIP("SO3part r must be on GPU");
    if(x.get_dev()!=1) GELIB_SKIP("SO3part x must be on GPU");
    if(y.get_dev()!=1) GELIB_SKIP("SO3part y must be on GPU");

    if(r.ndims()!=5) GELIB_SKIP("SO3part r must be 5D");
    if(x.ndims()!=5) GELIB_SKIP("SO3part x must be 5D");
    if(y.ndims()!=5) GELIB_SKIP("SO3part y must be 5D");

    auto rsubdims=r.dims.chunk(0,3);
    if(x.dims.chunk(0,3)!=rsubdims) GELIB_SKIP("leading dimensions of x and r must be same");
    if(y.dims.chunk(0,3)!=rsubdims) GELIB_SKIP("leading dimensions of y and r must be same");

    const int l1=x.getl();
    const int l2=y.getl();
    const int l=r.getl();
    const int L1=2*l1+1;
    const int L2=2*l2+1;
    if(l<std::abs(l1-l2) || l>l1+l2) GELIB_SKIP("|l1-l_2| <= l <= l1+l2 not satisfied");

    const int xn=x.dims[4];
    const int yn=y.dims[4];
    if(xn*yn+offs>r.dims[4]) GELIB_SKIP("fragment dimension of r not large enough");

    cnine::GPUtensor<float,5> rv(r);
    cnine::GPUtensor<float,5> xv(x);
    cnine::GPUtensor<float,5> yv(y);
    rv.arr+=r.strides[4]*offs;

    float* cptr=SO3_CGbank.get<float>(l1,l2,l,r.dev).get_arr(); 
    int clines=cnine::roundup(L1*L2,32)/32;

    bool tiled=xn*yn>1024;
    if(x.strides[3]!=x.strides[4]*x.dims[4]) tiled=true;
    if(y.strides[3]!=y.strides[4]*y.dims[4]) tiled=true;


    if(!tiled){ // Untiled option

      int nlines=cnine::roundup(L1*xn*2,32)/32+cnine::roundup(L2*yn*2,32)/32;

      if(nlines<=384){
	bool preloadCG=(nlines+clines<=384);
	dim3 blocks(r.dims[0],r.dims[1],r.dims[2]);
	SO3part_addCGproduct_kernel<<<blocks,cnine::roundup(xn*yn,32),(nlines+preloadCG*clines)*128,stream>>>
	  (rv,xv,yv,-1,cptr,preloadCG);
	return;
      }

    }else{ // Tiled option

      auto [xn,yn]=optimal_tile_size(x.getn(),y.getn());
      int nlines=cnine::roundup(L1*xn*2,32)/32+cnine::roundup(L2*yn*2,32)/32;

      if(nlines<=384){
	bool preloadCG=(nlines+clines<=384);
	dim3 blocks(r.dims[0],r.dims[1],r.dims[2]);
	SO3part_addCGproduct_tiled_kernel<<<blocks,cnine::roundup(xn*yn,32),(nlines+preloadCG*clines)*128,stream>>>
	  (rv,xv,yv,xn,yn,-1,cptr,preloadCG);
	return;
      }

    }

    GELIB_SKIP("no appropriate kernel found");

  }    


}


#endif 
