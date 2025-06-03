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


#ifndef _CnineCtensorConvolve3d_back0
#define _CnineCtensorConvolve3d_back0

#include "Ctensor6_view.hpp"
//#include "CSRmatrix.hpp"
#include "RtensorConvolve3d_back0.hpp"
#include "CtensorB.hpp"

namespace cnine{

  class CtensorConvolve3d_back0{
  public:

    void operator()(const Ctensor4_view& rg, const Ctensor4_view& xg, const Rtensor5_view& w){
      if(rg.s3!=1) {cnine_log.error(__PRETTY_FUNCTION__,"Last stride of rg must be 1. Skipping this operation."); return;}
      if(xg.s3!=1) {cnine_log.error(__PRETTY_FUNCTION__,"Last stride of xg must be 1. Skipping this operation."); return;}
      RtensorConvolve3d_back0()(rg.as_real().fuse34(),xg.as_real().fuse34(),w);
    }

    void operator()(const Ctensor5_view& rg, const Ctensor5_view& xg, const Rtensor5_view& w){
      if(rg.s4!=1) {cnine_log.error(__PRETTY_FUNCTION__,"Last stride of rg must be 1. Skipping this operation."); return;}
      if(xg.s4!=1) {cnine_log.error(__PRETTY_FUNCTION__,"Last stride of xg must be 1. Skipping this operation."); return;}
      RtensorConvolve3d_back0()(rg.as_real().fuse45(),xg.as_real().fuse45(),w);
    }

    void operator()(const Ctensor6_view& rg, const Ctensor6_view& xg, const Rtensor5_view& w){
      if(rg.s5!=1) {cnine_log.error(__PRETTY_FUNCTION__,"Last stride of rg must be 1. Skipping this operation."); return;}
      if(xg.s5!=1) {cnine_log.error(__PRETTY_FUNCTION__,"Last stride of xg must be 1. Skipping this operation."); return;}
      RtensorConvolve3d_back0()(rg.as_real().fuse56(),xg.as_real().fuse56(),w);
    }

  };

}

#endif 
