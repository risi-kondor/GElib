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
#include "CtensorEinsumFn.hpp"

using namespace cnine;

typedef RtensorA rtensor; 
typedef CtensorB ctensor; 


int main(int argc, char** argv){

  cnine_session session;

  cout<<endl;

  ctensor X=ctensor::gaussian({4,4});
  ctensor Y=ctensor::gaussian({4,4});
  ctensor R=ctensor::zeros({4,4});

  CtensorEinsumFn<float> fn("ij,ji->ab");
  fn(R.viewx(),X.viewx(),Y.viewx());

  PRINTL(R);

}
