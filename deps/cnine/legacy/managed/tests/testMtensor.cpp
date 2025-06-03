/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#include "Cnine_base.cpp"
#include "Mtensor.hpp"
#include "CnineSession.hpp"


using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;
  cout<<session<<endl;

  cout<<endl;

  Gdims dims({3,3});
  
  Mtensor<float> A=Mtensor<float>::sequential(dims);
  cout<<A<<endl;

  auto B=A+A;
  cout<<B<<endl;

  cout<<A*A<<endl;


  //Tensor<float> C(B);
  //cout<<C<<endl;

}

