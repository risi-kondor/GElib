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


#ifndef _CnineRtensorConvolve2d
#define _CnineRtensorConvolve2d

#include "Rtensor5_view.hpp"
#include "CSRmatrix.hpp"

namespace cnine{

  #ifdef _WITH_CUDA
  extern void RtensorConvolve2d_cu(const Rtensor4_view& r, const Rtensor4_view& x, const Rtensor4_view& w, const int padding0, const int padding1, const cudaStream_t& stream);
  extern void RtensorConvolve2d_cu(const Rtensor5_view& r, const Rtensor5_view& x, const Rtensor4_view& w, const int padding0, const int padding1, const cudaStream_t& stream);
  #endif


  class RtensorConvolve2d{
  public:


    // (i0,i1,a)*(a',j0,j1,a) -> (i0+j0,i1+j1,a') 
    void operator()(const Rtensor3_view& r, const Rtensor3_view& x, const Rtensor4_view& w){
      CNINE_CHECK_DEV3(r,x,w);
      CNINE_ASSRT(r.n2==w.n0);
      CNINE_ASSRT(x.n2==w.n3);
      int padding0=(r.n0-x.n0+w.n1-1)/2;
      int padding1=(r.n1-x.n1+w.n2-1)/2;

      if(r.dev==0){
	for(int i0=0; i0<r.n0; i0++){
	  for(int i1=0; i1<r.n1; i1++){
	    Rtensor1_view R(r.arr+i0*r.s0+i1*r.s1, r.n2, r.s2, r.dev);
	    for(int j0=std::max(0,padding0-i0); j0<std::min(w.n1,x.n0-i0+padding0); j0++){
		if(padding1==0){
		  Rtensor2_view W(w.arr+j0*w.s1, w.n0, w.n2*w.n3, w.s0, w.s3, w.dev);
		  Rtensor1_view X(x.arr+(i0+j0-padding0)*x.s0+i1*x.s1, w.n2*w.n3, x.s2, x.dev);
		  W.add_mprod_to(R,X);
		}else{
		  for(int j1=std::max(0,padding1-i1); j1<std::min(w.n2,x.n1-i1+padding1); j1++){
		    Rtensor2_view W(w.arr+j0*w.s1+j1*w.s2, w.n0, w.n3, w.s0, w.s3, w.dev);
		    Rtensor1_view X(x.arr+(i0+j0-padding0)*x.s0+(i1+j1-padding1)*x.s1, x.n2, x.s2, x.dev);
		    W.add_mprod_to(R,X);
		  }
		}
	    }
	  }
	}
      }
      if(r.dev==1){
	int dev=r.dev; CNINE_CPUONLY();
	//CUDA_STREAM(RtensorConvolve2d_cu(r,x,w,padding0,padding1,stream));
      }
    }


    // (i0,i1,a,c)*(a',j0,j1,a) -> (i0+j0,i1+j1,a',c) 
    void operator()(const Rtensor4_view& r, const Rtensor4_view& x, const Rtensor4_view& w){
      CNINE_CHECK_DEV3(r,x,w);
      CNINE_ASSRT(r.n2==w.n0);
      CNINE_ASSRT(x.n2==w.n3);
      CNINE_ASSRT(r.n3==x.n3);
      int padding0=(r.n0-x.n0+w.n1-1)/2;
      int padding1=(r.n1-x.n1+w.n2-1)/2;

      if(r.dev==0){
	for(int i0=0; i0<r.n0; i0++){
	  for(int i1=0; i1<r.n1; i1++){
	    Rtensor2_view R(r.arr+i0*r.s0+i1*r.s1, r.n2,r.n3, r.s2,r.s3, r.dev);
	    for(int j0=std::max(0,padding0-i0); j0<std::min(w.n1,x.n0-i0+padding0); j0++){
		if(padding1==0){
		  Rtensor2_view W(w.arr+j0*w.s1, w.n0, w.n2*w.n3, w.s0, w.s3, w.dev);
		  Rtensor2_view X(x.arr+(i0+j0-padding0)*x.s0+i1*x.s1, w.n2*x.n2, x.n3, x.s2, x.s3, x.dev);
		  R.add_mprod(W,X);
		}else{
		  for(int j1=std::max(0,padding1-i1); j1<std::min(w.n2,x.n1-i1+padding1); j1++){
		    Rtensor2_view W(w.arr+j0*w.s1+j1*w.s2, w.n0, w.n3, w.s0, w.s3, w.dev);
		    Rtensor2_view X(x.arr+(i0+j0-padding0)*x.s0+(i1+j1-padding1)*x.s1, 
		      x.n2, x.n3, x.s2, x.s3, x.dev);
		    R.add_mprod(W,X);
		  }
		}
	    }
	  }
	}
      }
      if(r.dev==1){
	//int dev=r.dev; CNINE_CPUONLY();
	CUDA_STREAM(RtensorConvolve2d_cu(r,x,w,padding0,padding1,stream));
      }
    }


    // (b,i0,i1,a,c)*(a',j0,j1,a) -> (b,i0+j0,i1+j1,a',c) 
    void operator()(const Rtensor5_view& r, const Rtensor5_view& x, const Rtensor4_view& w){
      CNINE_CHECK_DEV3(r,x,w);
      CNINE_ASSRT(r.n0==x.n0);
      CNINE_ASSRT(r.n3==w.n0);
      CNINE_ASSRT(r.n4==x.n4);
      CNINE_ASSRT(x.n3==w.n3);
      int padding0=(r.n1-x.n1+w.n1-1)/2;
      int padding1=(r.n2-x.n2+w.n2-1)/2;

      if(r.dev==0){
	for(int b=0; b<x.n0; b++){
	  (*this)(r.slice0(b),x.slice0(b),w);
	}
      }
      if(r.dev==1){
	CUDA_STREAM(RtensorConvolve2d_cu(r,x,w,padding0,padding1,stream));
      }
    }


    // (b,i0,i1,d,a,c)*(a',j0,j1,a) -> (b,i0+j0,i1+j1,d,a',c) 
    void operator()(const Rtensor6_view& r, const Rtensor6_view& x, const Rtensor4_view& w){
      CNINE_CHECK_DEV3(r,x,w);
      CNINE_ASSRT(r.n0==x.n0);
      CNINE_ASSRT(r.n3==x.n3);
      CNINE_ASSRT(r.n4==w.n0);
      CNINE_ASSRT(x.n4==w.n3);
      CNINE_ASSRT(r.n5==x.n5);
      int padding0=(r.n1-x.n1+w.n1-1)/2;
      int padding1=(r.n2-x.n2+w.n2-1)/2;

      if(r.dev==0){
	for(int b=0; b<x.n0; b++)
	  for(int d=0; d<x.n3; d++)
	    (*this)(r.slice0(b).slice2(d),x.slice0(b).slice2(d),w);
      }
      if(r.dev==1){
	int dev=r.dev; CNINE_CPUONLY();
	//CUDA_STREAM(RtensorConvolve2d_cu(r,x,w,padding0,padding1,stream));
      }
    }

  };






  inline RtensorA convolve2D(const RtensorA& x, const RtensorA& w, const int padding0=0, const int padding1=0){
      CNINE_ASSRT(w.ndims()==4);

      if(x.ndims()==3){
	CNINE_ASSRT(w.dims[3]==x.dims[2]);
	RtensorA r=RtensorA::zero({x.dims[0]+2*padding0-w.dims[1]+1,x.dims[1]+2*padding1-w.dims[2]+1,w.dims[0]},x.dev);
	RtensorConvolve2d()(r.view3(),x.view3(),w.view4());
	return r;
      }

      if(x.ndims()==4){ // add channels
	CNINE_ASSRT(w.dims[3]==x.dims[2]);
	RtensorA r=RtensorA::zero({x.dims[0]+2*padding0-w.dims[1]+1,x.dims[1]+2*padding1-w.dims[2]+1,w.dims[0],x.dims[3]},x.dev);
	RtensorConvolve2d()(r.view4(),x.view4(),w.view4());
	return r;
      }

      if(x.ndims()==5){ // add batches
	CNINE_ASSRT(w.dims[3]==x.dims[3]);
	RtensorA r=RtensorA::zero({x.dims[0],x.dims[1]+2*padding0-w.dims[1]+1,x.dims[2]+2*padding1-w.dims[2]+1,w.dims[0],x.dims[4]},x.dev);
	RtensorConvolve2d()(r.view5(),x.view5(),w.view4());
	return r;
      }

      if(x.ndims()==6){ // add blocks
	CNINE_ASSRT(w.dims[3]==x.dims[4]);
	RtensorA r=RtensorA::zero({x.dims[0],x.dims[1]+2*padding0-w.dims[1]+1,x.dims[2]+2*padding1-w.dims[2]+1,x.dims[3],w.dims[0],x.dims[5]},x.dev);
	RtensorConvolve2d()(r.view6(),x.view6(),w.view4());
	return r;
      }

      return RtensorA();
  }


}

#endif 



/*
  for(int i0=0; i0<r.n1; i0++)
  for(int i1=0; i1<r.n2; i1++){
  Rtensor2_view R(r.arr+b*r.s0+i0*r.s1+i1*r.s2, r.n3,r.n4, r.s3,r.s4, r.dev);
  for(int j0=std::max(0,padding0-i0); j0<std::min(w.n1,x.n1-i0+padding0); j0++){
  if(padding1==0){
  Rtensor2_view W(w.arr+j0*w.s1, w.n0, w.n2*w.n3, w.s0, w.s3, w.dev);
  Rtensor2_view X(x.arr+b*x.s0+(i0+j0-padding0)*x.s1+i1*x.s2, w.n2*x.n3,x.n4, x.s3,x.s4, x.dev);
  R.add_mprod(W,X);
  }else{
  for(int j1=std::max(0,padding1-i1); j1<std::min(w.n1,x.n2-i1+padding1); j1++){
  Rtensor2_view W(w.arr+j0*w.s1+j1*w.s2, w.n0, w.n3, w.s0, w.s3, w.dev);
  Rtensor2_view X(x.arr+b*x.s0+(i0+j0-padding0)*x.s1+(i1+j1-padding1)*x.s2, 
  x.n3, x.n4, x.s3, x.s4, x.dev);
  R.add_mprod(W,X);
  }
  }
  }
  }
*/

