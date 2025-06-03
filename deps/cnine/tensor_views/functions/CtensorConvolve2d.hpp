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


#ifndef _CnineCtensorConvolve2d
#define _CnineCtensorConvolve2d

#include "Ctensor5_view.hpp"
#include "CSRmatrix.hpp"
#include "RtensorConvolve2d.hpp"
#include "CtensorB.hpp"


namespace cnine{


  class CtensorConvolve2d{
  public:

    void operator()(const Ctensor3_view& r, const Ctensor3_view& x, const Rtensor4_view& w){
      RtensorConvolve2d()(r.as_real(),x.as_real(),w);
    }

    void operator()(const Ctensor4_view& r, const Ctensor4_view& x, const Rtensor4_view& w){
      if(r.s3!=1) {cnine_log.error(__PRETTY_FUNCTION__,"Last stride of r must be 1. Skipping this operation."); return;}
      if(x.s3!=1) {cnine_log.error(__PRETTY_FUNCTION__,"Last stride of x must be 1. Skipping this operation."); return;}
      RtensorConvolve2d()(r.as_real().fuse34(),x.as_real().fuse34(),w);
    }

    void operator()(const Ctensor5_view& r, const Ctensor5_view& x, const Rtensor4_view& w){
      if(r.s4!=1) {cnine_log.error(__PRETTY_FUNCTION__,"Last stride of r must be 1. Skipping this operation."); return;}
      if(x.s4!=1) {cnine_log.error(__PRETTY_FUNCTION__,"Last stride of x must be 1. Skipping this operation."); return;}
      RtensorConvolve2d()(r.as_real().fuse45(),x.as_real().fuse45(),w);
    }

    void operator()(const CtensorView& r, const CtensorView& x, const Rtensor4_view& w){
      if(r.ndims()==6){
	CNINE_ASSRT(x.ndims()==6);
	RtensorConvolve2d()(r.as_real().fuse(-1).view6(),x.as_real().fuse(-1).view6(),w);
      }
    }

  };



  inline CtensorB convolve2D(const CtensorB& x, const RtensorA& w, const int padding0=0, const int padding1=0){
      CNINE_ASSRT(w.ndims()==4);

      if(x.ndims()==3){
	CNINE_ASSRT(w.dims[3]==x.dims[2]);
	CtensorB r=CtensorB::zero({x.dims[0]+2*padding0-w.dims[1]+1,x.dims[1]+2*padding1-w.dims[2]+1,w.dims[0]});
	CtensorConvolve2d()(r.view3(),x.view3(),w.view4());
	return r;
      }

      if(x.ndims()==4){ // add channels
	CNINE_ASSRT(w.dims[3]==x.dims[2]);
	CtensorB r=CtensorB::zero({x.dims[0]+2*padding0-w.dims[1]+1,x.dims[1]+2*padding1-w.dims[2]+1,w.dims[0],x.dims[3]});
	CtensorConvolve2d()(r.view4(),x.view4(),w.view4());
	return r;
      }

      if(x.ndims()==5){ // add batches
	CNINE_ASSRT(w.dims[3]==x.dims[3]);
	CtensorB r=CtensorB::zero({x.dims[0],x.dims[1]+2*padding0-w.dims[1]+1,x.dims[2]+2*padding1-w.dims[2]+1,w.dims[0],x.dims[4]});
	CtensorConvolve2d()(r.view5(),x.view5(),w.view4());
	return r;
      }

      if(x.ndims()==6){ // add blocks
	CNINE_ASSRT(w.dims[3]==x.dims[4]);
	CtensorB r=CtensorB::zero({x.dims[0],x.dims[1]+2*padding0-w.dims[1]+1,x.dims[2]+2*padding1-w.dims[2]+1,x.dims[3],w.dims[0],x.dims[5]});
	CtensorConvolve2d()(r.viewx(),x.viewx(),w.view4());
	return r;
      }

      return CtensorB();
  }


}



#endif 
