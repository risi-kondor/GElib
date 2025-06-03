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


#ifndef _CnineCtensorConvolve3d
#define _CnineCtensorConvolve3d

#include "Ctensor6_view.hpp"
//#include "CSRmatrix.hpp"
#include "RtensorConvolve3d.hpp"
#include "CtensorB.hpp"


namespace cnine{


  class CtensorConvolve3d{
  public:

    void operator()(const Ctensor4_view& r, const Ctensor4_view& x, const Rtensor5_view& w){
      if(r.s3!=1) {cnine_log.error(__PRETTY_FUNCTION__,"Last stride of r must be 1. Skipping this operation."); return;}
      if(x.s3!=1) {cnine_log.error(__PRETTY_FUNCTION__,"Last stride of x must be 1. Skipping this operation."); return;}
      RtensorConvolve3d()(r.as_real().fuse34(),x.as_real().fuse34(),w);
    }

    void operator()(const Ctensor5_view& r, const Ctensor5_view& x, const Rtensor5_view& w){
      if(r.s4!=1) {cnine_log.error(__PRETTY_FUNCTION__,"Last stride of r must be 1. Skipping this operation."); return;}
      if(x.s4!=1) {cnine_log.error(__PRETTY_FUNCTION__,"Last stride of x must be 1. Skipping this operation."); return;}
      RtensorConvolve3d()(r.as_real().fuse45(),x.as_real().fuse45(),w);
    }

    void operator()(const Ctensor6_view& r, const Ctensor6_view& x, const Rtensor5_view& w){
      if(r.s5!=1) {cnine_log.error(__PRETTY_FUNCTION__,"Last stride of r must be 1. Skipping this operation."); return;}
      if(x.s5!=1) {cnine_log.error(__PRETTY_FUNCTION__,"Last stride of x must be 1. Skipping this operation."); return;}
      RtensorConvolve3d()(r.as_real().fuse56(),x.as_real().fuse56(),w);
    }

  };



  inline CtensorB convolve3D(const CtensorB& x, const RtensorA& w, const int padding0=0, const int padding1=0, const int padding2=0){
      CNINE_ASSRT(w.ndims()==4);

      if(x.ndims()==4){
	CNINE_ASSRT(w.dims[4]==x.dims[3]);
	CtensorB r=CtensorB::zero({x.dims[0]+2*padding0-w.dims[1]+1,x.dims[1]+2*padding1-w.dims[2]+1,
	      x.dims[2]+2*padding2-w.dims[3]+1,w.dims[0]},x.dev);
	CtensorConvolve3d()(r.view4(),x.view4(),w.view5());
	return r;
      }

      if(x.ndims()==5){ // add channels
	CNINE_ASSRT(w.dims[4]==x.dims[3]);
	CtensorB r=CtensorB::zero({x.dims[0]+2*padding0-w.dims[1]+1,x.dims[1]+2*padding1-w.dims[2]+1,
	      x.dims[2]+2*padding2-w.dims[3]+1,w.dims[0],x.dims[4]},x.dev);
	CtensorConvolve3d()(r.view5(),x.view5(),w.view5());
	return r;
      }

      if(x.ndims()==6){ // add batches
	CNINE_ASSRT(w.dims[4]==x.dims[4]);
	CtensorB r=CtensorB::zero({x.dims[0],x.dims[1]+2*padding0-w.dims[1]+1,x.dims[2]+2*padding1-w.dims[2]+1,
	      x.dims[3]+2*padding2-w.dims[3]+1,w.dims[0],x.dims[5]},x.dev);
	CtensorConvolve3d()(r.view6(),x.view6(),w.view5());
	return r;
      }

      //if(x.ndims()==6){ // add blocks
      //CNINE_ASSRT(w.dims[3]==x.dims[4]);
      //CtensorB r=CtensorB::zero({x.dims[0],x.dims[1]+2*padding0-w.dims[1]+1,x.dims[2]+2*padding1-w.dims[2]+1,x.dims[3],w.dims[0],x.dims[5]});
      //CtensorConvolve3d()(r.viewx(),x.viewx(),w.view4());
      //return r;
      //}

      return CtensorB();
  }


}



#endif 
