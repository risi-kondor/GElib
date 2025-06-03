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


#include "Cnine_base.cpp"
#include "CtensorB.hpp"
#include "RtensorA.hpp"
#include "IntTensor.hpp"
#include "CnineSession.hpp"
#include "ScaleSomeSlicesFn.hpp"

using namespace cnine;

typedef RtensorA rtensor; 
typedef CtensorB ctensor; 


int main(int argc, char** argv){

  cnine_session session;

  cout<<endl;

  rtensor A=rtensor::sequential({5,5});

  rtensor c(Gdims({2}));
  c.set(0,2);
  c.set(1,3);

  IntTensor ix=IntTensor::zero(Gdims({2}));
  ix.set(0,1);
  ix.set(1,2);

  ScaleSomeSlicesFn()(A.view2(),ix.view1(),c.view1());
  
  PRINTL(A);

}
  
