/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineRtensorArrayFunctions
#define _CnineRtensorArrayFunctions

#include "Cnine_base.hpp"
#include "ExprTemplates.hpp"
#include "CscalarObj.hpp"
#include "RtensorObj.hpp"
#include "RtensorArray.hpp"


namespace cnine{


  // ---- Broadcast operations ------------------------------------------------------------------------------ 


  //RtensorArray broadcast(const Gdims& adims, const RtensorObj& x){
  //return RtensorArray(adims,x);
  //}

  
  RtensorArray operator*(const RtensorArray& x, const Broadcast<RtensorObj>& _y){
    const RtensorObj& y=_y.obj;
    RtensorArray R(x.adims,x.cdims.Mprod(y.dims),x.nbu,fill::zero,x.dev);
    R.broadcast_add_mprod(x,y);
    return R;
  }




}

#endif 
