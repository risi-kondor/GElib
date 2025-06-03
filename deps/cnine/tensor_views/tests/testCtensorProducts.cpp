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
#include "CnineSession.hpp"

using namespace cnine;

typedef CtensorB ctensor; 


int main(int argc, char** argv){

  cnine_session session;

  cout<<endl;

  ctensor A=ctensor::sequential(Gdims(3,3,3));
  ctensor B=ctensor::sequential(Gdims(3,3,3));
  ctensor C=ctensor::zero(Gdims(3,3,3,3));

  C.view4().add_expand_2(A.view3(),B.view3());
  print(C);


  #ifdef _WITH_CUDA

  ctensor Ac=A.to_device(1);
  ctensor Bc=B.to_device(1);
  ctensor Cc=ctensor::zero(Gdims(3,3,3,3),1);
  
  Cc.view4().add_expand_2(Ac.view3(),Bc.view3());
  print(Cc);

  #endif 


  ctensor D=ctensor::zero(Gdims(3,3));
  
  A.view3().add_contract_aib_aib_ab_to(D.view2(),B.view3());
  print(D);


  #ifdef _WITH_CUDA

  ctensor Dc=ctensor::zero(Gdims(3,3),1);
  Ac.view3().add_contract_aib_aib_ab_to(Dc.view2(),Bc.view3());
  print(Dc);

  #endif
}
