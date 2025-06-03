/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2022, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineRtensorConvolve2dSparse
#define _CnineRtensorConvolve2dSparse

#include "Rtensor5_view.hpp"
#include "CSRmatrix.hpp"

namespace cnine{

  #ifdef _WITH_CUDA
  extern void RtensorConvolve2d_cu(const Rtensor5_view& r, const Rtensor5_view& x, const CSRmatrix<float>& w, const int padding0, const int padding1, const cudaStream_t& stream);
  #endif


  class RtensorConvolve2dSparse{
  public:


    void operator()(const Rtensor3_view& r, const Rtensor3_view& x, const CSRmatrix<float>& w, const int J0, const int J1){
      CNINE_CHECK_DEV3(r,x,w);
      CNINE_ASSRT(r.n2==w.n);
      int padding0=(r.n0-x.n0+J0-1)/2;
      int padding1=(r.n1-x.n1+J1-1)/2;
      //const int J1=r.n1-x.n1+1-2*padding1;
      const int A=x.n2;

      if(r.dev==0){
	for(int i0=0; i0<r.n0; i0++)
	  for(int i1=0; i1<r.n1; i1++){
	    Rtensor1_view R(r.arr+i0*r.s0+i1*r.s1, r.n2, r.s2, r.dev);
	      w.for_each([&](const int aout, const int s, const float v){
		  int j0=s/(J1*A);
		  int j1=(s/A)%J1;
		  if(i0+j0-padding0<0 || i0+j0-padding0>=x.n0) return;
		  if(i1+j1-padding1<0 || i1+j1-padding1>=x.n1) return;
		  int a=s%A;
		  R.inc(aout,v*x(i0+j0-padding0,i1+j1-padding1,a));
		});
	  }
	}
      if(r.dev==1){
	int dev=r.dev; CNINE_CPUONLY();
	//CUDA_STREAM(RtensorConvolve2d_cu(r,x,w,padding0,padding1,stream));
      }
    }


    void operator()(const Rtensor4_view& r, const Rtensor4_view& x, const CSRmatrix<float>& w, const int J0, const int J1){
      CNINE_CHECK_DEV3(r,x,w);
      CNINE_ASSRT(r.n2==w.n);
      CNINE_ASSRT(r.n3==x.n3);
      int padding0=(r.n0-x.n0+J0-1)/2;
      int padding1=(r.n1-x.n1+J1-1)/2;
      //const int J0=r.n0-x.n0+1-2*padding0;
      //const int J1=r.n1-x.n1+1-2*padding1;
      const int A=x.n2;
      const int C=x.n3;

      if(r.dev==0){
	for(int i0=0; i0<r.n0; i0++)
	  for(int i1=0; i1<r.n1; i1++){
	    Rtensor2_view R(r.arr+i0*r.s0+i1*r.s1, r.n2, r.n3, r.s2, r.s3, r.dev);
	      w.for_each([&](const int aout, const int s, const float v){
		  int j0=s/(J1*A);
		  int j1=(s/A)%J1;
		  if(i0+j0-padding0<0 || i0+j0-padding0>=x.n0) return;
		  if(i1+j1-padding1<0 || i1+j1-padding1>=x.n1) return;
		  int a=s%A;
		  for(int c=0; c<C; c++)
		    R.inc(aout,c,v*x(i0+j0-padding0,i1+j1-padding1,a,c));
		});
	  }
      }
      if(r.dev==1){
	int dev=r.dev; CNINE_CPUONLY();
	//CUDA_STREAM(RtensorConvolve2d_cu(r,x,w,padding0,padding1,stream));
      }
    }


    void operator()(const Rtensor5_view& r, const Rtensor5_view& x, const CSRmatrix<float>& w, const int J0, const int J1){
      CNINE_CHECK_DEV3(r,x,w);
      CNINE_ASSRT(r.n0==x.n0);
      CNINE_ASSRT(r.n3==w.n);
      CNINE_ASSRT(r.n4==x.n4);
      int padding0=(r.n1-x.n1+J0-1)/2;
      int padding1=(r.n2-x.n2+J1-1)/2;

      if(r.dev==0){
	for(int b=0; b<x.n0; b++)
	  (*this)(r.slice0(b),x.slice0(b),w,J0,J1);
      }
      if(r.dev==1){
	int dev=r.dev; CNINE_CPUONLY();
	//CUDA_STREAM(RtensorConvolve2d_cu(r,x,w,padding0,padding1,stream));
      }
    }


    void operator()(const Rtensor6_view& r, const Rtensor6_view& x, const CSRmatrix<float>& w, const int J0, const int J1){
      CNINE_CHECK_DEV3(r,x,w);
      CNINE_ASSRT(r.n0==x.n0);
      CNINE_ASSRT(r.n4==w.n);
      CNINE_ASSRT(r.n3==x.n3);
      CNINE_ASSRT(r.n5==x.n5);
      int padding0=(r.n1-x.n1+J0-1)/2;
      int padding1=(r.n2-x.n2+J1-1)/2;

      if(r.dev==0){
	for(int b=0; b<x.n0; b++)
	  for(int d=0; d<x.n3; d++)
	    (*this)(r.slice0(b).slice2(d),x.slice0(b).slice2(d),w,J0,J1);
      }
      if(r.dev==1){
	int dev=r.dev; CNINE_CPUONLY();
	//CUDA_STREAM(RtensorConvolve2d_cu(r,x,w,padding0,padding1,stream));
      }
    }


  };



  inline RtensorA convolve2D(const RtensorA& x, const CSRmatrix<float>& w, const int J0, const int J1, const int padding0=0, const int padding1=0){

      if(x.ndims()==3){
	CNINE_ASSRT(w.m==J0*J1*x.dims[2]);
	RtensorA r=RtensorA::zero({x.dims[0]+2*padding0-J0+1,x.dims[1]+2*padding1-J1+1,w.n});
	RtensorConvolve2dSparse()(r.view3(),x.view3(),w,J0,J1);
	return r;
      }

      if(x.ndims()==4){ // add channels
	CNINE_ASSRT(w.m==J0*J1*x.dims[2]);
	RtensorA r=RtensorA::zero({x.dims[0]+2*padding0-J0+1,x.dims[1]+2*padding1-J1+1,w.n,x.dims[3]});
	RtensorConvolve2dSparse()(r.view4(),x.view4(),w,J0,J1);
	return r;
      }

      if(x.ndims()==5){ // add batches
	CNINE_ASSRT(w.m==J0*J1*x.dims[3]);
	RtensorA r=RtensorA::zero({x.dims[0],x.dims[1]+2*padding0-J0+1,x.dims[2]+2*padding1-J1+1,w.n,x.dims[4]});
	RtensorConvolve2dSparse()(r.view5(),x.view5(),w,J0,J1);
	return r;
      }

      if(x.ndims()==6){ // add blocks
	CNINE_ASSRT(w.m==J0*J1*x.dims[4]);
	RtensorA r=RtensorA::zero({x.dims[0],x.dims[1]+2*padding0-J0+1,x.dims[2]+2*padding1-J1+1,x.dims[3],w.n,x.dims[5]});
	RtensorConvolve2dSparse()(r.view6(),x.view6(),w,J0,J1);
	return r;
      }

      return RtensorA();
  }

}

#endif 
