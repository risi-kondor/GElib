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


#ifndef _CnineCtensorArrayFunctions
#define _CnineCtensorArrayFunctions

#include "Cnine_base.hpp"
#include "ExprTemplates.hpp"
#include "CscalarObj.hpp"
#include "CtensorObj.hpp"
#include "CtensorArray.hpp"


namespace cnine{


  // ---- Broadcast operations ------------------------------------------------------------------------------ 


  //CtensorArray broadcast(const Gdims& adims, const CtensorObj& x){
  //return CtensorArray(adims,x);
  //}

  
  CtensorArray operator*(const CtensorArray& x, const Broadcast<CtensorObj>& _y){
    const CtensorObj& y=_y.obj;
    CtensorArray R(x.get_adims(),x.get_cdims().Mprod(y.dims),fill::zero,x.dev);
    R.broadcast_add_mprod(x,y);
    return R;
  }




}

#endif 
