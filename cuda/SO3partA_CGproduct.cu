
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3partA_CGproduct_cu
#define _SO3partA_CGproduct_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

//__device__ __constant__ unsigned char cg_cmem[32276]; 


#include "SO3partA.hpp"
#include "SO3partArrayA.hpp"
#include "SO3_CGbank.hpp"

#include "CellwiseBinaryCmap.hpp"
#include "BroadcastBinaryCmap.hpp"
#include "InnerCmap.hpp"
#include "OuterCmap.hpp"
#include "MVprodCmap.hpp"
#include "VMprodCmap.hpp"
//#include "convolve1_cmap.hpp"
#include "Convolve2Cmap.hpp"

extern GElib::SO3_CGbank SO3_cgbank;


__device__ void SO3part_load_lines(float* dest, const float* source, const int nlines, const int t){
  if(t<32){
    for(int i=0; i<nlines; i++)
      dest[i*32+t]=source[i*32+t];
  }
}

__device__ void SO3part_save_lines(const float* source, float* dest, const int nlines, const int t){
  if(t<32){
    for(int i=0; i<nlines; i++)
      dest[i*32+t]=source[i*32+t];
  }
}


// ---- CGproduct --------------------------------------------------------------------------------------------


template<typename IMAP>
__global__ void SO3partA_CGproduct_kernel(float* rarr, float* rarrc, float* xarr, float* xarrc, 
  float* yarr, float* yarrc, const int rstride, const int xstride, const int ystride, const IMAP cmap, 
  const int xn, const int yn, const int rn, const int l1, const int l2, const int l, 
  const int _offs, const int nch, const int Cptr, const int mode=0){

  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);

  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int t=threadIdx.x;

  const int r=2*l+1;
  const int r1=2*l1+1;
  const int r2=2*l2+1;

  const int xwidth=xn*nch; 
  const int ywidth=yn*nch; 
  const int rwidth=xn*yn*nch;
  const int global_rwidth=rn*nch;
  
  const int rlines=((r*rwidth-1)/32+1);
  const int xlines=((r1*xwidth-1)/32+1);
  const int ylines=((r2*ywidth-1)/32+1);

  const int rptr=0;
  const int xptr=rptr+rlines*64;
  const int yptr=xptr+xlines*64;

  int rix,xix,yix;
  int nsum;
  int lst;


  if(mode<2){
    auto T=cmap(blockIdx.x,blockIdx.y,blockIdx.z);
    rix=thrust::get<0>(T);
    xix=thrust::get<1>(T);
    yix=thrust::get<2>(T);
    nsum=1;
    //if(t==0) printf("foop1\n");
  }else{
    rix=cmap.target(blockIdx.x);
    nsum=cmap.n_accum(blockIdx.x);
    lst=cmap.lst_ptr(blockIdx.x);
  }
  
  if(mode==1){
    if(t<32){
      for(int i=0; i<2*rlines; i++)
	shared[rptr+i*32+t]=0;
    }
  }else{
    if(t<rwidth){
      for(int i=0; i<r; i++)
	shared[rptr+i*rwidth+t]=rarr[rix*rstride+_offs+i*global_rwidth+t];
      for(int i=0; i<r; i++)
	shared[rptr+rlines*32+i*rwidth+t]=rarrc[rix*rstride+_offs+i*global_rwidth+t];
    }
  }
  
  for(int s=0; s<nsum; s++){

    if(mode==2){
      auto T=cmap.source(lst,blockIdx.x,s);
      xix=thrust::get<0>(T);
      yix=thrust::get<1>(T);
    }

    SO3part_load_lines(shared+xptr,xarr+xix*xstride,xlines,t);
    SO3part_load_lines(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);
    SO3part_load_lines(shared+yptr,yarr+yix*ystride,ylines,t);
    SO3part_load_lines(shared+yptr+ylines*32,yarrc+yix*ystride,ylines,t);

    //if(t==0) printf("foop3\n");
      
      __syncthreads();

      const int rpr=rptr+t;
      const int rpi=rpr+rlines*32;

      const int xcol=t/yn;
      const int xpr=xptr+xcol;
      const int xpi=xpr+xlines*32;

      const int ycol=t%ywidth;
      const int ypr=yptr+ycol;
      const int ypi=ypr+ylines*32;


      if(t<rwidth){
	for(int m1=-l1; m1<=l1; m1++){
	  const float x_r=shared[xpr+xwidth*(m1+l1)];
	  const float x_i=shared[xpi+xwidth*(m1+l1)];
	  int lower=-l-m1; if(lower<-l2) lower=-l2;
	  int upper=l-m1; if(upper>l2) upper=l2;
	  for(int m2=lower; m2<=upper; m2++){
	    float c=C_ptr[(m1+l1)*r2+m2+l2];
	    const float y_r=shared[ypr+ywidth*(m2+l2)];
	    const float y_i=shared[ypi+ywidth*(m2+l2)];
	    shared[rpr+rwidth*(m1+m2+l)]+=c*(x_r*y_r-x_i*y_i); 
	    shared[rpi+rwidth*(m1+m2+l)]+=c*(x_r*y_i+x_i*y_r);
	  }
	}
      }

    //if(t==0) printf("foop4\n");

      __syncthreads();
  }

  //if(t==0) printf("fooq\n");
  
  if(t<rwidth){
    for(int i=0; i<r; i++)
      rarr[rix*rstride+_offs+i*global_rwidth+t]=shared[rptr+i*rwidth+t];
    for(int i=0; i<r; i++)
      rarrc[rix*rstride+_offs+i*global_rwidth+t]=shared[rptr+rlines*32+i*rwidth+t];
  }    
 
}



template<typename IMAP>
__global__ void SO3partA_CGproduct_kernel_L(float* rarr, float* rarrc, float* xarr, float* xarrc, 
  float* yarr, float* yarrc, const int rstride, const int xstride, const int ystride, const IMAP cmap, 
  const int xn, const int yn, const int rn, const int l1, const int l2, const int l, 
  const int _offs, const int nch, const int Cptr, const int mode=0){

  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);

  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int t=threadIdx.x;

  const int r=2*l+1;
  const int r1=2*l1+1;
  const int r2=2*l2+1;

  const int xwidth=xn*nch; 
  const int ywidth=yn*nch; 
  const int rwidth=xn*nch; 
  const int global_rwidth=rn*nch;
  
  const int rlines=((r*rwidth-1)/32+1);
  const int xlines=((r1*xwidth-1)/32+1);
  const int ylines=((r2*1-1)/32+1);

  const int rptr=0;
  const int xptr=rptr+rlines*64;
  const int yptr=xptr+xlines*64;

  int rix,xix,yix;
  int nsum;
  int lst;

  if(mode<2){
    auto T=cmap(blockIdx.x,blockIdx.y,blockIdx.z);
    rix=thrust::get<0>(T);
    xix=thrust::get<1>(T);
    yix=thrust::get<2>(T);
    nsum=1;
  }else{
    rix=cmap.target(blockIdx.x);
    nsum=cmap.n_accum(blockIdx.x);
    lst=cmap.lst_ptr(blockIdx.x);
  }
  

  for(int s=0; s<nsum; s++){

    if(mode==2){
      auto T=cmap.source(lst,blockIdx.x,s);
      xix=thrust::get<0>(T);
      yix=thrust::get<1>(T);
    }

    SO3part_load_lines(shared+xptr,xarr+xix*xstride,xlines,t);
    SO3part_load_lines(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);


    for(int ycol=0; ycol<ywidth; ycol++){

      if(t==0){
	for(int i=0; i<r; i++)
	  shared[yptr+i]=yarr[yix*ystride+i*ywidth+ycol];
	for(int i=0; i<r; i++)
	  shared[yptr+ylines*32+i]=yarrc[yix*ystride+i*ywidth+ycol];
      }

      if(t<32){
	for(int i=0; i<2*rlines; i++)
	  shared[rptr+i*32+t]=0;
      }
      
      __syncthreads();

      const int rpr=rptr+t;
      const int rpi=rpr+rlines*32;

      const int xcol=t;
      const int xpr=xptr+xcol;
      const int xpi=xpr+xlines*32;

      //const int ycol=t%ywidth;
      const int ypr=yptr;// +ycol;
      const int ypi=ypr+ylines*32;


      if(t<rwidth){
	for(int m1=-l1; m1<=l1; m1++){
	  const float x_r=shared[xpr+xwidth*(m1+l1)];
	  const float x_i=shared[xpi+xwidth*(m1+l1)];
	  int lower=-l-m1; if(lower<-l2) lower=-l2;
	  int upper=l-m1; if(upper>l2) upper=l2;
	  for(int m2=lower; m2<=upper; m2++){
	    float c=C_ptr[(m1+l1)*r2+m2+l2];
	    const float y_r=shared[ypr+1*(m2+l2)];
	    const float y_i=shared[ypi+1*(m2+l2)];
	    shared[rpr+rwidth*(m1+m2+l)]+=c*(x_r*y_r-x_i*y_i); 
	    shared[rpi+rwidth*(m1+m2+l)]+=c*(x_r*y_i+x_i*y_r);
	  }
	}
      }

      //if(t==0) printf("foop4\n");

      __syncthreads();

      //if(t==0) printf("fooq\n");
  
      if(t<rwidth){
	for(int i=0; i<r; i++)
	  rarr[rix*rstride+_offs+i*global_rwidth+t*ywidth+ycol]+=shared[rptr+i*rwidth+t];
	for(int i=0; i<r; i++)
	  rarrc[rix*rstride+_offs+i*global_rwidth+t*ywidth+ycol]+=shared[rptr+rlines*32+i*rwidth+t];
      }    

      __syncthreads();

    } // ycol

  } //nsum
}




// ---- back0 ------------------------------------------------------------------------------------------------


template<typename IMAP>
__global__ void SO3partA_CGproduct_back0_kernel(float* xarr, float* xarrc, float* garr, float* garrc, 
  float* yarr, float* yarrc, const int xstride, const int ystride, const int gstride, const IMAP cmap, 
  const int xn, const int yn, const int gn, const int l1, const int l2, const int l, 
  const int _offs, const int nch, const int Cptr, const int mode=0){

  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);

  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int t=threadIdx.x;

  const int rg=2*l+1;
  const int rx=2*l1+1;
  const int ry=2*l2+1;

  const int xwidth=xn*nch; 
  const int ywidth=yn*nch; 
  const int gwidth=xn*yn*nch;
  const int global_gwidth=gn*nch;

  const int glines=((rg*gwidth-1)/32+1);
  const int xlines=((rx*xwidth-1)/32+1);
  const int ylines=((ry*ywidth-1)/32+1);

  const int xptr=0;
  const int gptr=xptr+xlines*64;
  const int yptr=gptr+glines*64;

  int gix,xix,yix;
  int nsum;
  int lst;

  if(mode<2){
    auto T=cmap(blockIdx.x,blockIdx.y,blockIdx.z);
    xix=thrust::get<0>(T);
    gix=thrust::get<1>(T);
    yix=thrust::get<2>(T);
    nsum=1;
  }else{
    xix=cmap.target(blockIdx.x);
    nsum=cmap.n_accum(blockIdx.x);
    lst=cmap.lst_ptr(blockIdx.x);
  }

  if(mode==1){
    if(t<32){
      for(int i=0; i<2*xlines; i++){
	shared[xptr+i*32+t]=0;
      }
    }
  }else{
    SO3part_load_lines(shared+xptr,xarr+xix*xstride,xlines,t);
    SO3part_load_lines(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);
  }
  
  for(int s=0; s<nsum; s++){

    if(mode==2){
      auto T=cmap.source(lst,blockIdx.x,s);
      gix=thrust::get<0>(T);
      yix=thrust::get<1>(T);
    }

    // hack: gwidth assumed to be <=32
    for(int i=0; i<rg; i++)
      if(t<gwidth)
	shared[gptr+i*gwidth+t]=garr[gix*gstride+i*global_gwidth+_offs+t];
    for(int i=0; i<rg; i++)
      if(t<gwidth)
	shared[gptr+glines*32+i*gwidth+t]=garrc[gix*gstride+i*global_gwidth+_offs+t];

    SO3part_load_lines(shared+yptr,yarr+yix*ystride,ylines,t);
    SO3part_load_lines(shared+yptr+ylines*32,yarrc+yix*ystride,ylines,t);

    __syncthreads();

      //const int xcol=t;
    const int xpr=xptr+t;
    const int xpi=xpr+xlines*32;
    
    for(int ycol=0; ycol<ywidth; ycol++){
      
      const int ypr=yptr+ycol;
      const int ypi=ypr+ylines*32;
      
      const int gpr=gptr+ywidth*t+ycol;
      const int gpi=gpr+glines*32;

      if(t<xwidth){
	for(int m1=-l1; m1<=l1; m1++){
	  int lower=-l-m1; if(lower<-l2) lower=-l2;
	  int upper=l-m1; if(upper>l2) upper=l2;
	  for(int m2=lower; m2<=upper; m2++){
	    float c=C_ptr[(m1+l1)*ry+m2+l2];
	    const float y_r=shared[ypr+ywidth*(m2+l2)];
	    const float y_i=shared[ypi+ywidth*(m2+l2)];
	    const float g_r=shared[gpr+gwidth*(m1+m2+l)];
	    const float g_i=shared[gpi+gwidth*(m1+m2+l)];
	    shared[xpr+xwidth*(m1+l1)]+=c*(g_r*y_r+g_i*y_i); 
	    shared[xpi+xwidth*(m1+l1)]+=c*(-g_r*y_i+g_i*y_r);
	  }
	}
      }
      __syncthreads();
	
    }

  }
  
  SO3part_save_lines(shared+xptr,xarr+xix*xstride,xlines,t);
  SO3part_save_lines(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);
    
  __syncthreads();

}


template<typename IMAP>
__global__ void SO3partA_CGproduct_back0_kernel_big(float* xarr, float* xarrc, float* garr, float* garrc, 
  float* yarr, float* yarrc, const int xstride, const int ystride, const int gstride, const IMAP cmap, 
  const int xn, const int yn, const int gn, const int l1, const int l2, const int l, 
  const int _offs, const int nch, const int Cptr, const int mode=0){

  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);

  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int t=threadIdx.x;

  //const int rg=2*l+1;
  const int rx=2*l1+1;
  const int ry=2*l2+1;

  const int xwidth=xn*nch; 
  const int ywidth=yn*nch; 
  //const int gwidth=xn*yn*nch;
  const int global_gwidth=gn*nch;

  //const int glines=((rg*gwidth-1)/32+1);
  const int xlines=((rx*xwidth-1)/32+1);
  const int ylines=((ry*ywidth-1)/32+1);

  const int xptr=0;
  const int yptr=xptr+xlines*64;
  //const int yptr=gptr+glines*64;

  int gix,xix,yix;
  int nsum;
  int lst;

  if(mode<2){
    auto T=cmap(blockIdx.x,blockIdx.y,blockIdx.z);
    xix=thrust::get<0>(T);
    gix=thrust::get<1>(T);
    yix=thrust::get<2>(T);
    nsum=1;
  }else{
    xix=cmap.target(blockIdx.x);
    nsum=cmap.n_accum(blockIdx.x);
    lst=cmap.lst_ptr(blockIdx.x);
  }

  if(mode==1){
    if(t<32){
      for(int i=0; i<2*xlines; i++){
	shared[xptr+i*32+t]=0;
      }
    }
  }else{
    SO3part_load_lines(shared+xptr,xarr+xix*xstride,xlines,t);
    SO3part_load_lines(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);
  }
  
  for(int s=0; s<nsum; s++){

    if(mode==2){
      auto T=cmap.source(lst,blockIdx.x,s);
      gix=thrust::get<0>(T);
      yix=thrust::get<1>(T);
    }

    // hack: gwidth assumed to be <=32
    //for(int i=0; i<rg; i++)
    //if(t<gwidth)
    //shared[gptr+i*gwidth+t]=garr[gix*gstride+i*global_gwidth+_offs+t];
    //for(int i=0; i<rg; i++)
    //if(t<gwidth)
    //shared[gptr+glines*32+i*gwidth+t]=garrc[gix*gstride+i*global_gwidth+_offs+t];

    SO3part_load_lines(shared+yptr,yarr+yix*ystride,ylines,t);
    SO3part_load_lines(shared+yptr+ylines*32,yarrc+yix*ystride,ylines,t);

    __syncthreads();

      //const int xcol=t;
    const int xpr=xptr+t;
    const int xpi=xpr+xlines*32;
    
    for(int ycol=0; ycol<ywidth; ycol++){
      
      const int ypr=yptr+ycol;
      const int ypi=ypr+ylines*32;
      
      //const int gpr=gptr+ywidth*t+ycol;
      //const int gpi=gpr+glines*32;

      if(t<xwidth){
	for(int m1=-l1; m1<=l1; m1++){
	  int lower=-l-m1; if(lower<-l2) lower=-l2;
	  int upper=l-m1; if(upper>l2) upper=l2;
	  for(int m2=lower; m2<=upper; m2++){
	    float c=C_ptr[(m1+l1)*ry+m2+l2];
	    const float y_r=shared[ypr+ywidth*(m2+l2)];
	    const float y_i=shared[ypi+ywidth*(m2+l2)];
	    //const float g_r=shared[gpr+gwidth*(m1+m2+l)];
	    //const float g_i=shared[gpi+gwidth*(m1+m2+l)];
	    const float g_r=garr[gix*gstride+_offs+ywidth*t+ycol+(m1+m2+l)*global_gwidth];
	    const float g_i=garrc[gix*gstride+_offs+ywidth*t+ycol+(m1+m2+l)*global_gwidth];
	    shared[xpr+xwidth*(m1+l1)]+=c*(g_r*y_r+g_i*y_i); 
	    shared[xpi+xwidth*(m1+l1)]+=c*(-g_r*y_i+g_i*y_r);
	  }
	}
      }
      __syncthreads();
	
    }

  }
  
  SO3part_save_lines(shared+xptr,xarr+xix*xstride,xlines,t);
  SO3part_save_lines(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);
    
  __syncthreads();

}


// ---- back1 ------------------------------------------------------------------------------------------------


template<typename IMAP>
__global__ void SO3partA_CGproduct_back1_kernel(float* yarr, float* yarrc, float* garr, float* garrc, 
  float* xarr, float* xarrc, const int xstride, const int ystride, const int gstride, const IMAP cmap, 
  const int xn, const int yn, const int gn, const int l1, const int l2, const int l, 
  const int _offs, const int nch, const int Cptr, const int mode=0){

  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);

  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int t=threadIdx.x;

  const int rg=2*l+1;
  const int rx=2*l1+1;
  const int ry=2*l2+1;

  const int xwidth=xn*nch; 
  const int ywidth=yn*nch; 
  const int gwidth=xn*yn*nch;
  const int global_gwidth=gn*nch;

  const int glines=((rg*gwidth-1)/32+1);
  const int xlines=((rx*xwidth-1)/32+1);
  const int ylines=((ry*ywidth-1)/32+1);

  const int yptr=0;
  const int gptr=yptr+ylines*64;
  const int xptr=gptr+glines*64;

  int gix,xix,yix;
  int nsum;
  int lst;

  if(mode<2){
    auto T=cmap(blockIdx.x,blockIdx.y,blockIdx.z);
    yix=thrust::get<0>(T);
    gix=thrust::get<1>(T);
    xix=thrust::get<2>(T);
    nsum=1;
  }else{
    yix=cmap.target(blockIdx.x);
    nsum=cmap.n_accum(blockIdx.x);
    lst=cmap.lst_ptr(blockIdx.x);
  }

  if(mode==1){
    if(t<32){
      for(int i=0; i<2*ylines; i++)
	shared[yptr+i*32+t]=0;
    }
  }else{
    SO3part_load_lines(shared+yptr,yarr+yix*ystride,ylines,t);
    SO3part_load_lines(shared+yptr+ylines*32,yarrc+yix*ystride,ylines,t);
  }
  
  for(int s=0; s<nsum; s++){

    if(mode==2){
      auto T=cmap.source(lst,blockIdx.x,s);
      gix=thrust::get<0>(T);
      xix=thrust::get<1>(T);
    }

    // hack: gwidth assumed to be <=32
    for(int i=0; i<rg; i++)
      if(t<gwidth)
	shared[gptr+i*gwidth+t]=garr[gix*gstride+i*global_gwidth+_offs+t];
    for(int i=0; i<rg; i++)
      if(t<gwidth)
	shared[gptr+glines*32+i*gwidth+t]=garrc[gix*gstride+i*global_gwidth+_offs+t];

    SO3part_load_lines(shared+xptr,xarr+xix*xstride,xlines,t);
    SO3part_load_lines(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);

    __syncthreads();

    //const int ycol=t;
    const int ypr=yptr+t;
    const int ypi=ypr+ylines*32;
    
    for(int xcol=0; xcol<xwidth; xcol++){
      
      const int xpr=xptr+xcol;
      const int xpi=xpr+xlines*32;
      
      const int gpr=gptr+ywidth*xcol+t;
      const int gpi=gpr+glines*32;

      if(t<ywidth){
	for(int m1=-l1; m1<=l1; m1++){
	  int lower=-l-m1; if(lower<-l2) lower=-l2;
	  int upper=l-m1; if(upper>l2) upper=l2;
	  const float x_r=shared[xpr+xwidth*(m1+l1)];
	  const float x_i=shared[xpi+xwidth*(m1+l1)];
	  for(int m2=lower; m2<=upper; m2++){
	    float c=C_ptr[(m1+l1)*ry+m2+l2];
	    const float g_r=shared[gpr+gwidth*(m1+m2+l)];
	    const float g_i=shared[gpi+gwidth*(m1+m2+l)];
	    shared[ypr+ywidth*(m2+l2)]+=c*(g_r*x_r+g_i*x_i); 
	    shared[ypi+ywidth*(m2+l2)]+=c*(-g_r*x_i+g_i*x_r);
	  }
	}
      }
      __syncthreads();

    }

  }
  
  SO3part_save_lines(shared+yptr,yarr+yix*ystride,ylines,t);
  SO3part_save_lines(shared+yptr+ylines*32,yarrc+yix*ystride,ylines,t);
    
  __syncthreads();

}


template<typename IMAP>
__global__ void SO3partA_CGproduct_back1_kernel_big(float* yarr, float* yarrc, float* garr, float* garrc, 
  float* xarr, float* xarrc, const int xstride, const int ystride, const int gstride, const IMAP cmap, 
  const int xn, const int yn, const int gn, const int l1, const int l2, const int l, 
  const int _offs, const int nch, const int Cptr, const int mode=0){

  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);

  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int t=threadIdx.x;

  //const int rg=2*l+1;
  const int rx=2*l1+1;
  const int ry=2*l2+1;

  const int xwidth=xn*nch; 
  const int ywidth=yn*nch; 
  //const int gwidth=xn*yn*nch;
  const int global_gwidth=gn*nch;

  //const int glines=((rg*gwidth-1)/32+1);
  const int xlines=((rx*xwidth-1)/32+1);
  const int ylines=((ry*ywidth-1)/32+1);

  const int yptr=0;
  const int xptr=yptr+ylines*64;
  //const int xptr=gptr+glines*64;

  int gix,xix,yix;
  int nsum;
  int lst;

  if(mode<2){
    auto T=cmap(blockIdx.x,blockIdx.y,blockIdx.z);
    yix=thrust::get<0>(T);
    gix=thrust::get<1>(T);
    xix=thrust::get<2>(T);
    nsum=1;
  }else{
    yix=cmap.target(blockIdx.x);
    nsum=cmap.n_accum(blockIdx.x);
    lst=cmap.lst_ptr(blockIdx.x);
  }

  if(mode==1){
    if(t<32){
      for(int i=0; i<2*ylines; i++)
	shared[yptr+i*32+t]=0;
    }
  }else{
    SO3part_load_lines(shared+yptr,yarr+yix*ystride,ylines,t);
    SO3part_load_lines(shared+yptr+ylines*32,yarrc+yix*ystride,ylines,t);
  }
  
  for(int s=0; s<nsum; s++){

    if(mode==2){
      auto T=cmap.source(lst,blockIdx.x,s);
      gix=thrust::get<0>(T);
      xix=thrust::get<1>(T);
    }

    // hack: gwidth assumed to be <=32
    //for(int i=0; i<rg; i++)
    //if(t<gwidth)
    //shared[gptr+i*gwidth+t]=garr[gix*gstride+i*global_gwidth+_offs+t];
    //for(int i=0; i<rg; i++)
    //if(t<gwidth)
    //shared[gptr+glines*32+i*gwidth+t]=garrc[gix*gstride+i*global_gwidth+_offs+t];

    SO3part_load_lines(shared+xptr,xarr+xix*xstride,xlines,t);
    SO3part_load_lines(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);

    __syncthreads();

    //const int ycol=t;
    const int ypr=yptr+t;
    const int ypi=ypr+ylines*32;
    
    for(int xcol=0; xcol<xwidth; xcol++){
      
      const int xpr=xptr+xcol;
      const int xpi=xpr+xlines*32;
      
      //const int gpr=gptr+ywidth*xcol+t;
      //const int gpi=gpr+glines*32;

      if(t<ywidth){
	for(int m1=-l1; m1<=l1; m1++){
	  int lower=-l-m1; if(lower<-l2) lower=-l2;
	  int upper=l-m1; if(upper>l2) upper=l2;
	  const float x_r=shared[xpr+xwidth*(m1+l1)];
	  const float x_i=shared[xpi+xwidth*(m1+l1)];
	  for(int m2=lower; m2<=upper; m2++){
	    float c=C_ptr[(m1+l1)*ry+m2+l2];
	    //const float g_r=shared[gpr+gwidth*(m1+m2+l)];
	    //const float g_i=shared[gpi+gwidth*(m1+m2+l)];
	    const float g_r=garr[gix*gstride+_offs+ywidth*xcol+t+(m1+m2+l)*global_gwidth];
	    const float g_i=garrc[gix*gstride+_offs+ywidth*xcol+t+(m1+m2+l)*global_gwidth];
	    shared[ypr+ywidth*(m2+l2)]+=c*(g_r*x_r+g_i*x_i); 
	    shared[ypi+ywidth*(m2+l2)]+=c*(-g_r*x_i+g_i*x_r);
	  }
	}
      }
      __syncthreads();

    }

  }
  
  SO3part_save_lines(shared+yptr,yarr+yix*ystride,ylines,t);
  SO3part_save_lines(shared+yptr+ylines*32,yarrc+yix*ystride,ylines,t);
    
  __syncthreads();

}


// -----------------------------------------------------------------------------------------------------------


namespace GElib{


  template<typename CMAP>
  void SO3partA_CGproduct_cu(const CMAP& map, SO3partArrayA& r, const SO3partArrayA& x, 
    const SO3partArrayA& y, const cudaStream_t& stream, const int offs, const int mode){

    const int xl=x.getl();
    const int yl=y.getl();
    const int l=r.getl();
    const int _nch=1;
    assert(x.nbu==r.nbu);
    assert(y.nbu==r.nbu);
    int _nbu=1; if(_nbu<0) _nbu=1;

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    //int nlines=x.cellstride/16+y.cellstride/16+r.cellstride/16; // should be smaller than this!
    int nlines=x.cellstride/16+y.cellstride/16+cnine::roundup(x.getn()*y.getn()*_nch*(2*l+1),32)/16;
    // nlines/=_nbu;

    cout<<"nlines="<<nlines<<endl;

    if(nlines<=0*384){

      SO3partA_CGproduct_kernel<<<map.blockdims(),cnine::roundup(x.getn()*y.getn(),32),nlines*128,stream>>>
	(r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,
	  r.cellstride,x.cellstride,y.cellstride,map,
	  x.getn(),y.getn(),r.getn(),xl,yl,l,offs,_nch,Cptr,mode);

    }else{
      
      int nlines=x.cellstride/16+cnine::roundup(_nch*(2*yl+1),32)/16+cnine::roundup(x.getn()*_nch*(2*l+1),32)/16;

      cout<<"GElib: large CGproduct"<<endl; 

      if(nlines>384){
	cout<<"GElib error: CGproduct too big for shared memory"<<endl;
      }else{
	SO3partA_CGproduct_kernel_L<<<map.blockdims(),cnine::roundup(x.getn(),32),nlines*128,stream>>>
	  (r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,
	    r.cellstride,x.cellstride,y.cellstride,map,
	    x.getn(),y.getn(),r.getn(),xl,yl,l,offs,_nch,Cptr,mode);
      }
    }

  }

  
  void SO3partA_CGproduct_cu(SO3partA& r, const SO3partA& x, const SO3partA& y,  const int offs, 
    const cudaStream_t& stream,const int mode){

    const int xl=x.getl();
    const int yl=y.getl();
    const int l=r.getl();
    const int _nch=1;
    assert(x.nbu==r.nbu);
    assert(y.nbu==r.nbu);
    int _nbu=1; if(_nbu<0) _nbu=1;
    cnine::CellwiseBinaryCmap map;

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    int nlines=cnine::roundup(x.memsize,32)/32+cnine::roundup(y.memsize,32)/32+
      cnine::roundup(x.getn()*y.getn()*_nch*(2*l+1),32)/16;

    //cout<<"nlines="<<nlines<<endl;

    if(nlines<=384){

      SO3partA_CGproduct_kernel<<<map.blockdims(),cnine::roundup(x.getn()*y.getn(),32),nlines*128,stream>>>
	(r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,
	  0,0,0,map,
	  x.getn(),y.getn(),r.getn(),xl,yl,l,offs,_nch,Cptr,mode);

    }else{
      
      int nlines=cnine::roundup(x.memsize,32)/32+cnine::roundup(y.memsize,32)/32+
        cnine::roundup(x.getn()*_nch*(2*l+1),32)/16;

      cout<<"GElib: large CGproduct"<<endl; 

      if(nlines>384){
	cout<<"GElib error: CGproduct too big for shared memory"<<endl;
      }else{
	SO3partA_CGproduct_kernel_L<<<map.blockdims(),cnine::roundup(x.getn(),32),nlines*128,stream>>>
	  (r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,
	    0,0,0,map,
	    x.getn(),y.getn(),r.getn(),xl,yl,l,offs,_nch,Cptr,mode);
      }
    }

  }

  
  template<typename CMAP>
  void SO3partA_CGproduct_back0_cu(const CMAP& map, SO3partArrayA& x, const SO3partArrayA& g, 
    const SO3partArrayA& y, const cudaStream_t& stream, const int offs, const int mode){

    const int xl=x.getl();
    const int yl=y.getl();
    const int l=g.getl();

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    int nlines=x.cellstride/16+y.cellstride/16+g.cellstride/16;
    assert(x.nbu==g.nbu);
    assert(y.nbu==g.nbu);

    const int _nch=1;
    int _nbu=1; if(_nbu<0) _nbu=1;
    nlines/=_nbu;

    cout<<"nlines="<<nlines<<endl;

    if(nlines<=0*384){

      SO3partA_CGproduct_back0_kernel<<<map.blockdims(),cnine::roundup(x.getn(),32),nlines*128,stream>>>
	(x.arrg,x.arrgc,g.arrg,g.arrgc,y.arrg,y.arrgc,
	  x.cellstride,y.cellstride,g.cellstride,map,
	  x.getn(),y.getn(),g.getn(),xl,yl,l,offs,_nch,Cptr,mode);
      
    }else{

      int nlines=x.cellstride/16+y.cellstride/16;
      
      cout<<"GElib: large CGproduct_back0"<<endl; 

      if(nlines>384){
	cout<<"GElib error: CGproduct too big for shared memory"<<endl;
      }else{
	SO3partA_CGproduct_back0_kernel_big<<<map.blockdims(),cnine::roundup(std::max(x.getn(),y.getn()),32),nlines*128,stream>>>
	  (x.arrg,x.arrgc,g.arrg,g.arrgc,y.arrg,y.arrgc,
	    x.cellstride,y.cellstride,g.cellstride,map,
	    x.getn(),y.getn(),g.getn(),xl,yl,l,offs,_nch,Cptr,mode);
      }

    }

  }
  

  template<typename CMAP>
  void SO3partA_CGproduct_back1_cu(const CMAP& map, SO3partArrayA& y, const SO3partArrayA& g, 
    const SO3partArrayA& x, const cudaStream_t& stream, const int offs, const int mode){

    const int xl=x.getl();
    const int yl=y.getl();
    const int l=g.getl();

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    int nlines=x.cellstride/16+y.cellstride/16+g.cellstride/16;
    assert(x.nbu==g.nbu);
    assert(y.nbu==g.nbu);

    const int _nch=1;
    int _nbu=1; if(_nbu<0) _nbu=1;
    nlines/=_nbu;

    cout<<"nlines="<<nlines<<endl;

    if(nlines<=0*384){

      SO3partA_CGproduct_back1_kernel<<<map.blockdims(),cnine::roundup(y.getn(),32),nlines*128,stream>>>
	(y.arrg,y.arrgc,g.arrg,g.arrgc,x.arrg,x.arrgc,
	  x.cellstride,y.cellstride,g.cellstride,map,
	  x.getn(),y.getn(),g.getn(),xl,yl,l,offs,_nch,Cptr,mode);
      
    }else{

      int nlines=x.cellstride/16+y.cellstride/16;
      
      cout<<"GElib: large CGproduct_back1"<<endl; 

      if(nlines>384){
	cout<<"GElib error: CGproduct too big for shared memory"<<endl;
      }else{
	SO3partA_CGproduct_back1_kernel_big<<<map.blockdims(),cnine::roundup(std::max(x.getn(),y.getn()),32),nlines*128,stream>>>
	  (y.arrg,y.arrgc,g.arrg,g.arrgc,x.arrg,x.arrgc,
	    x.cellstride,y.cellstride,g.cellstride,map,
	    x.getn(),y.getn(),g.getn(),xl,yl,l,offs,_nch,Cptr,mode);
      }

    }

  }


  template void SO3partA_CGproduct_cu(const cnine::CellwiseBinaryCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_CGproduct_cu(const cnine::BroadcastBinaryCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_CGproduct_cu(const cnine::OuterCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_CGproduct_cu(const cnine::InnerCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_CGproduct_cu(const cnine::MVprodCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_CGproduct_cu(const cnine::Convolve2Cmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);



  template void SO3partA_CGproduct_back0_cu(const cnine::CellwiseBinaryCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_CGproduct_back0_cu(const cnine::BroadcastBinaryCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_CGproduct_back0_cu(const cnine::OuterCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);


  template void SO3partA_CGproduct_back1_cu(const cnine::CellwiseBinaryCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_CGproduct_back1_cu(const cnine::BroadcastBinaryCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_CGproduct_back1_cu(const cnine::OuterCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);




}

#endif 





  /*
  void SO3partA_CGproduct_cu(SO3partArrayA& r, const SO3partArrayA& x, const SO3partArrayA& y, 
    const int mode, const cudaStream_t& stream, const int offs){

    const int xl=x.getl();
    const int yl=y.getl();
    const int l=r.getl();

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    int nlines=x.cellstride/16+y.cellstride/16+r.cellstride/16;
    assert(x.nbu==r.nbu);
    assert(y.nbu==r.nbu);

    const int _nch=1;
    int _nbu=1; if(_nbu<0) _nbu=1;
    nlines/=_nbu;

    if(mode==0){
      dim3 blocks(r.aasize,1,1);
      cnine::CellwiseImap imap;
      SO3partA_CGproduct_kernel<<<blocks,cnine::roundup(x.getn()*y.getn(),32),nlines*128,stream>>>
	(r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,
	  r.cellstride,x.cellstride,y.cellstride,imap,
	  x.getn(),y.getn(),r.getn(),xl,yl,l,offs,_nch,Cptr); // 
    }

    if(mode==1){
      dim3 blocks(x.aasize,y.aasize,1);
      cnine::OuterImap imap(r.adims[1]);
      SO3partA_CGproduct_kernel<<<blocks,cnine::roundup(x.getn()*y.getn(),32),nlines*128,stream>>>
	(r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,
	  r.cellstride,x.cellstride,y.cellstride,imap,
	  x.getn(),y.getn(),r.getn(),xl,yl,l,offs,_nch,Cptr);
    }
    

  }
    */


  /*
  void SO3partA_CGproduct_cu(SO3partArrayA& r, const SO3partArrayA& x, const SO3partArrayA& y, 
    const int rN, const int xN, const int yN, 
    const int ris, const int rjs, const int rks, 
    const int xis, const int xjs, const int xks, 
    const int yis, const int yjs, const int yks, 
    const cudaStream_t& stream, const int offs){

    const int xl=x.getl();
    const int yl=y.getl();
    const int l=r.getl();

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    int nlines=x.cellstride/16+y.cellstride/16+r.cellstride/16;
    assert(x.nbu==r.nbu);
    assert(y.nbu==r.nbu);

    const int _nch=1;
    int _nbu=1; if(_nbu<0) _nbu=1;
    dim3 blocks(rN,xN,yN);
    nlines/=_nbu;


    SO3partA_CGproduct_kernel<<<blocks,cnine::roundup(x.getn()*y.getn(),32),nlines*128,stream>>>
      (r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,
	ris*r.cellstride,rjs*r.cellstride, rks*r.cellstride,
	xis*x.cellstride,xjs*x.cellstride, xks*x.cellstride,
	yis*y.cellstride,yjs*y.cellstride, yks*y.cellstride,
	x.getn(),y.getn(),r.getn(),xl,yl,l,offs,_nch,Cptr); 
  }
  */

/*
__global__ void SO3partA_CGproduct_kernel(float* rarr, float* rarrc, float* xarr, float* xarrc, 
  float* yarr, float* yarrc, 
  const int ristride, const int xistride, const int yistride,   
  const int rjstride, const int xjstride, const int yjstride,   
  const int rkstride, const int xkstride, const int ykstride,   
  const int xfrags, const int yfrags, const int rfrags,  
  const int l1, const int l2, const int l, const int _offs, const int nch, const int Cptr){

  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);

  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;

  const int iix=blockIdx.x;
  const int jix=blockIdx.y;
  const int kix=blockIdx.z;

  const int t=threadIdx.x;

  const int r1=2*l1+1;
  const int r2=2*l2+1;
  const int r=2*l+1;

  const int xwidth=xfrags*nch; 
  const int ywidth=yfrags*nch; 
  const int rwidth=xfrags*yfrags*nch;
  const int global_rwidth=rfrags*nch;

  int offs=0;

  int xptr=32*offs;
  SO3part_load(offs,shared,xarr,xarrc,l1,xwidth,iix*xistride+jix*xjstride+kix*xkstride,t);

  const int yptr=32*offs;
  SO3part_load(offs,shared,yarr,yarrc,l2,ywidth,iix*yistride+jix*yjstride+kix*ykstride,t);

  const int rpr=32*offs+t;
  const int rpi=rpr+((r*rwidth-1)/32+1)*32;
  float* _rptr=rarr+iix*ristride+jix*rjstride+kix*rkstride+_offs;
  float* _rptri=rarrc+iix*ristride+jix*rjstride+kix*rkstride+_offs;

  if(t<rwidth){
    for(int i=0; i<r; i++)
      shared[rpr+i*rwidth]=_rptr[i*global_rwidth+t];
    for(int i=0; i<r; i++)
      shared[rpi+i*rwidth]=_rptri[i*global_rwidth+t];
  }

  __syncthreads();
  
  const int xcol=t/yfrags;
  const int xpr=xptr+xcol;
  const int xlines=((r1*xwidth-1)/32+1); 
  const int xpi=xpr+xlines*32;

  const int ycol=t%ywidth;
  const int ypr=yptr+ycol;
  const int ylines=((r2*ywidth-1)/32+1); 
  const int ypi=ypr+ylines*32;

  if(t<rwidth){
    for(int m1=-l1; m1<=l1; m1++){
      const float x_r=shared[xpr+xwidth*(m1+l1)];
      const float x_i=shared[xpi+xwidth*(m1+l1)];
      int lower=-l-m1; if(lower<-l2) lower=-l2;
      int upper=l-m1; if(upper>l2) upper=l2;
      for(int m2=lower; m2<=upper; m2++){
	float c=C_ptr[(m1+l1)*r2+m2+l2];
	const float y_r=shared[ypr+ywidth*(m2+l2)];
	const float y_i=shared[ypi+ywidth*(m2+l2)];
	shared[rpr+rwidth*(m1+m2+l)]+=c*(x_r*y_r-x_i*y_i); 
	shared[rpi+rwidth*(m1+m2+l)]+=c*(x_r*y_i+x_i*y_r);
      }
    }
  }

  __syncthreads();

  if(t<rwidth){
    for(int i=0; i<r; i++)
      _rptr[i*global_rwidth+t]=shared[rpr+i*rwidth];
    for(int i=0; i<r; i++)
      _rptri[i*global_rwidth+t]=shared[rpi+i*rwidth];
  }

}
*/

/*
__device__ int SO3part_load(int& offs, float* shared, float* arr, float* arrc, const int l, const int w, const int skip, const int t){
  const int _offs=offs;
  int ptr=32*offs;
  const int r=2*l+1;
  const int lines=((r*w-1)/32+1); 
  float* xcell=arr+skip; 
  if(t<32)
    for(int i=0; i<lines; i++)
      shared[ptr+i*32+t]=xcell[i*32+t];
  ptr+=32*lines;
  xcell=arrc+skip;
  if(t<32)
    for(int i=0; i<lines; i++)
      shared[ptr+i*32+t]=xcell[i*32+t];
  offs+=2*lines;
  return 32*_offs;
}


__device__ int SO3part_load(int& offs, float* shared, float* arr, float* arrc, const int l, const int w, const int skip, 
  const int frag_offs, const int nfrags, const int t){
  const int _offs=offs;
  int ptr=32*offs+t;
  const int r=2*l+1;
  const int lines=((r*w-1)/32+1); 
  float* xcell=arr+skip+frag_offs+t;
  if(t<nfrags){
    for(int i=0; i<r; i++)
      shared[ptr+i*nfrags]=xcell[i*w];
    ptr+=32*lines;
    xcell=arrc+skip+frag_offs+t;
    for(int i=0; i<r; i++)
      shared[ptr+i*nfrags]=xcell[i*w];
  }
  offs+=2*lines;
  return 32*_offs;
}
*/

