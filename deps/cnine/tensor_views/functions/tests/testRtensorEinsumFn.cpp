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
#include "CnineSession.hpp"
#include "RtensorEinsumFn.hpp"

using namespace cnine;

typedef RtensorA rtensor; 


int main(int argc, char** argv){

  cnine_session session;

  cout<<endl;

  rtensor X=rtensor::sequential({4,4});
  rtensor Y=rtensor::sequential({4,4});
  rtensor R=rtensor::zeros({4,4});

  RtensorEinsumFn<float> fn("ij,ji->ab");
  fn(R.viewx(),X.viewx(),Y.viewx());

  PRINTL(R);

}
