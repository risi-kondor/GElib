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


#ifndef _CnineRtensorConvolve3d_back0
#define _CnineRtensorConvolve3d_back0

#include "RtensorConvolve3d.hpp"

namespace cnine{

  class RtensorConvolve3d_back0{
  public:

    // forward: (i0,i1,i2,a)*(a',j0,j1,j2,a) -> (i0+j0,i1+j1,i2+j2,a') 
    void operator()(const Rtensor4_view& rg, const Rtensor4_view& xg, const Rtensor5_view& w){
      Rtensor5_view wt(w.arr+(w.n1-1)*w.s1+(w.n2-1)*w.s2+(w.n3-1)*w.s3,
	w.n4,w.n1,w.n2,w.n3,w.n0,
	w.s4,-w.s1,-w.s2,-w.s3,w.s0,w.dev);
      RtensorConvolve3d()(xg,rg,wt);
    }
    
    // forward: (i0,i1,i2,a,c)*(a',j0,j1,j2,a) -> (i0+j0,i1+j1,i2+j2,a',c) 
    void operator()(const Rtensor5_view& rg, const Rtensor5_view& xg, const Rtensor5_view& w){
      Rtensor5_view wt(w.arr+(w.n1-1)*w.s1+(w.n2-1)*w.s2+(w.n3-1)*w.s3,
	w.n4,w.n1,w.n2,w.n3,w.n0,
	w.s4,-w.s1,-w.s2,-w.s3,w.s0,w.dev);
      RtensorConvolve3d()(xg,rg,wt);
    }

    // forward: (b,i0,i1,i2,a,c)*(a',j0,j1,j2,a) -> (b,i0+j0,i1+j1,i2+j2,a',c) 
    void operator()(const Rtensor6_view& rg, const Rtensor6_view& xg, const Rtensor5_view& w){
      Rtensor5_view wt(w.arr+(w.n1-1)*w.s1+(w.n2-1)*w.s2+(w.n3-1)*w.s3,
      w.n4,w.n1,w.n2,w.n3,w.n0,
      w.s4,-w.s1,-w.s2,-w.s3,w.s0,w.dev);
      RtensorConvolve3d()(xg,rg,wt);
    }

  };

}

#endif 
