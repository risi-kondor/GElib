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


#ifndef _CnineRtensorConvolve3d
#define _CnineRtensorConvolve3d

#include "Rtensor6_view.hpp"
#include "LoggedOp.hpp"

namespace cnine{

  #ifdef _WITH_CUDA
  extern void RtensorConvolve3d_cu(const Rtensor4_view& r, const Rtensor4_view& x, const Rtensor5_view& w, const int padding0, const int padding1, const int padding2, const cudaStream_t& stream);
  extern void RtensorConvolve3d_cu(const Rtensor5_view& r, const Rtensor5_view& x, const Rtensor5_view& w, const int padding0, const int padding1, const int padding2, const cudaStream_t& stream);
  extern void RtensorConvolve3d_cu(const Rtensor6_view& r, const Rtensor6_view& x, const Rtensor5_view& w, const int padding0, const int padding1, const int padding2, const cudaStream_t& stream);
  #endif


  class RtensorConvolve3d{
  public:


    // (i0,i1,i2,a)*(a',j0,j1,j2,a) -> (i0+j0,i1+j1,i2+j2,a') 
    void operator()(const Rtensor4_view& r, const Rtensor4_view& x, const Rtensor5_view& w){
      CNINE_CHECK_DEV3(r,x,w);
      CNINE_ASSRT(r.n3==w.n0);
      CNINE_ASSRT(x.n3==w.n4);
      int padding0=(r.n0-x.n0+w.n1-1)/2;
      int padding1=(r.n1-x.n1+w.n2-1)/2;
      int padding2=(r.n2-x.n2+w.n3-1)/2;

      if(r.dev==0){
	for(int i0=0; i0<r.n0; i0++){
	  for(int i1=0; i1<r.n1; i1++){
	    for(int i2=0; i2<r.n2; i2++){
	      Rtensor1_view R(r.arr+i0*r.s0+i1*r.s1+i2*r.s2, r.n3, r.s3, r.dev);
	      for(int j0=std::max(0,padding0-i0); j0<std::min(w.n1,x.n0-i0+padding0); j0++){
		for(int j1=std::max(0,padding1-i1); j1<std::min(w.n2,x.n1-i1+padding1); j1++){
		  if(padding2==0){
		    Rtensor2_view W(w.arr+j0*w.s1+j1*w.s2, w.n0, w.n3*w.n4, w.s0, w.s4, w.dev); // should be Rtensor2_view?
		    Rtensor1_view X(x.arr+(i0+j0-padding0)*x.s0+(i1+j1-padding1)*x.s1+i2*x.s2, w.n3*w.n4, x.s3, x.dev);
		    W.add_mprod_to(R,X);
		  }else{
		    for(int j2=std::max(0,padding2-i2); j2<std::min(w.n2,x.n2-i2+padding2); j2++){
		      Rtensor2_view W(w.arr+j0*w.s1+j1*w.s2+j2*w.s3, w.n0, w.n4, w.s0, w.s4, w.dev);
		      Rtensor1_view X(x.arr+(i0+j0-padding0)*x.s0+(i1+j1-padding1)*x.s1+(i2+j2-padding2)*x.s2, x.n3, x.s3, x.dev);
		      W.add_mprod_to(R,X);
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
      if(r.dev==1){
	//int dev=r.dev; CNINE_CPUONLY();
	CUDA_STREAM(RtensorConvolve3d_cu(r,x,w,padding0,padding1,padding2,stream));
      }
    }


    // (i0,i1,i2,a,c)*(a',j0,j1,j2,a) -> (i0+j0,i1+j1,i2+j2,a',c) 
    void operator()(const Rtensor5_view& r, const Rtensor5_view& x, const Rtensor5_view& w){
      CNINE_CHECK_DEV3(r,x,w);
      CNINE_ASSRT(r.n3==w.n0);
      CNINE_ASSRT(x.n3==w.n4);
      CNINE_ASSRT(r.n4==x.n4);
      int padding0=(r.n0-x.n0+w.n1-1)/2;
      int padding1=(r.n1-x.n1+w.n2-1)/2;
      int padding2=(r.n2-x.n2+w.n3-1)/2;

      if(r.dev==0){
	for(int i0=0; i0<r.n0; i0++){
	  for(int i1=0; i1<r.n1; i1++){
	    for(int i2=0; i2<r.n2; i2++){
	      Rtensor2_view R(r.arr+i0*r.s0+i1*r.s1+i2*r.s2,r.n3,r.n4,r.s3,r.s4,r.dev);
	      for(int j0=std::max(0,padding0-i0); j0<std::min(w.n1,x.n0-i0+padding0); j0++){
		for(int j1=std::max(0,padding1-i1); j1<std::min(w.n2,x.n1-i1+padding1); j1++){
		  for(int j2=std::max(0,padding2-i2); j2<std::min(w.n3,x.n2-i2+padding2); j2++){
		    Rtensor2_view W(w.arr+j0*w.s1+j1*w.s2+j2*w.s3, w.n0, w.n4, w.s0, w.s4, w.dev);
		    Rtensor2_view X(x.arr+(i0+j0-padding0)*x.s0+(i1+j1-padding1)*x.s1+(i2+j2-padding2)*x.s2, 
		      x.n3, x.n4, x.s3, x.s4, x.dev);
		    R.add_mprod(W,X);
		  }
		}
	      }
	      //cout<<i0<<" "<<i1<<" "<<i2<<" "<<R;
	    }
	  }
	}
      }
      if(r.dev==1){
	CUDA_STREAM(RtensorConvolve3d_cu(r,x,w,padding0,padding1,padding2,stream));
      }
    }


    // (b,i0,i1,i2,a,c)*(a',j0,j1,j2,a) -> (b,i0+j0,i1+j1,i2+j2,a',c) 
    void operator()(const Rtensor6_view& r, const Rtensor6_view& x, const Rtensor5_view& w){
      CNINE_CHECK_DEV3(r,x,w);
      CNINE_ASSRT(r.n0==x.n0);
      CNINE_ASSRT(r.n4==w.n0);
      CNINE_ASSRT(r.n5==x.n5);
      CNINE_ASSRT(x.n4==w.n4);
      int padding0=(r.n1-x.n1+w.n1-1)/2;
      int padding1=(r.n2-x.n2+w.n2-1)/2;
      int padding2=(r.n3-x.n3+w.n3-1)/2;

      //cout<<"RtensorConvolve3d "<<r.repr()<<" "<<x.repr()<<" "<<w.repr()<<endl;
      LoggedOp("RtensorConvolve3d",r,x,w);

      if(r.dev==0){
	for(int b=0; b<x.n0; b++){
	  (*this)(r.slice0(b),x.slice0(b),w);
	}
      }

      if(r.dev==1){
	CUDA_STREAM(RtensorConvolve3d_cu(r,x,w,padding0,padding1,padding2,stream));
      }
    }



  };


  inline RtensorA convolve3D(const RtensorA& x, const RtensorA& w, const int padding0=0, const int padding1=0, const int padding2=0){
      CNINE_ASSRT(w.ndims()==5);

      if(x.ndims()==4){
	CNINE_ASSRT(w.dims[4]==x.dims[3]);
	RtensorA r=RtensorA::zero({x.dims[0]+2*padding0-w.dims[1]+1,x.dims[1]+2*padding1-w.dims[2]+1,
	      x.dims[2]+2*padding2-w.dims[3]+1,w.dims[0]},x.dev);
	RtensorConvolve3d()(r.view4(),x.view4(),w.view5());
	return r;
      }

      if(x.ndims()==5){ // add channels
	CNINE_ASSRT(w.dims[4]==x.dims[3]);
	RtensorA r=RtensorA::zero({x.dims[0]+2*padding0-w.dims[1]+1,x.dims[1]+2*padding1-w.dims[2]+1,
	      x.dims[2]+2*padding2-w.dims[3]+1,w.dims[0],x.dims[4]},x.dev);
	RtensorConvolve3d()(r.view5(),x.view5(),w.view5());
	return r;
      }

      if(x.ndims()==6){ // add batches
	CNINE_ASSRT(w.dims[4]==x.dims[4]);
	RtensorA r=RtensorA::zero({x.dims[0],x.dims[1]+2*padding0-w.dims[1]+1,x.dims[2]+2*padding1-w.dims[2]+1,
	      x.dims[3]+2*padding2-w.dims[3]+1,w.dims[0],x.dims[5]},x.dev);
	RtensorConvolve3d()(r.view6(),x.view6(),w.view5());
	return r;
      }

      return RtensorA();
  }


}

#endif 
