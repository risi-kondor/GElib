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


#ifndef _CnineRtensorConvolve3dFn
#define _CnineRtensorConvolve3dFn

#include "Rtensor6_View.hpp"

namespace cnine{

  #ifdef _WITH_CUDA
  //extern void RtensorConvolve3d_cu(const Rtensor6_view& r, const Rtensor6_view& x, const Rtensor5_view& w, , const cudaStream_t& stream);
  #endif


  // (b,i0,i1,i2,a,c)*(a',j0,j1,j2,a) ->(b,i0+j0,i1+j1,i2+j2,a',c) 
  class RtensorConvolve3dFn{
  public:

    void operator()(const Rtensor6_view& r, const Rtensor6_view& x, const Rtensor5_view& w){
      CNINE_CHECK_DEV3(r,x,w);
      CNINE_ASSRT(r.n0==x.n0);
      CNINE_ASSRT(r.n1==x.n1-w.n1+1);
      CNINE_ASSRT(r.n2==x.n2-w.n2+1);
      CNINE_ASSRT(r.n3==x.n3-w.n3+1);
      CNINE_ASSRT(r.n4==w.n0);
      CNINE_ASSRT(r.n5==x.n5);
      CNINE_ASSRT(x.n3==w.n3);

      if(r.dev==0){
	for(int b=0; b<x.n0; b++)
	  for(int i0=0; i0<r.n1; i0++)
	    for(int i1=0; i1<r.n2; i1++)
	      for(int i2=0; i1<r.n3; i2++){
		Rtensor2_view R(r.arr+b*r.s0+i0*r.s1+i1*r.s2+i2*r.s3, r.n4,r.n5, r.s4,r.s5, r.dev);
		for(int j0=0; j0<w.n1; j0++){
		  for(int j1=0; j1<w.n2; j1++){
		    Rtensor2_view W(w.arr+j0*w.s1+j2*w.s2, w.n0,w.n3*w.n4, w.s0,w.s4, w.dev);
		    Rtensor2_view X(x.arr+b*x.s0+(i0+j0)*x.s1+(i1+j1)*x.s2+i2*x.s3, w.n3*x.n4,x.n5, x.s4,x.s5, x.dev);
		    R.add_mprod(W,X);
		  }
		}
	      }
      }
      if(r.dev==1){
	//CUDA_STREAM(RtensorConvolve3d_cu(r,x,w,stream));
      }
    }

  };

}

#endif

