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
#include "Tensor.hpp"
#include "TensorFunctions.hpp"
#include "CnineSession.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  cout<<endl;

  Gdims dims({3,3});
  
  Tensor<float> A=Tensor<float>::sequential(dims);
  cout<<A<<endl;

  auto B=A.slice(1,1);
  cout<<B<<endl;

  B.set(1,99);
  cout<<B<<endl;
  cout<<A<<endl;

  A.slice(1,0)=A.slice(0,2);
  cout<<A<<endl;

  Tensor<float> C(A);
  C.set(0,0,32);
  cout<<C<<endl;
  cout<<A<<endl;

  cout<<A+A<<endl;


  auto U=Tensor<float>::random_unitary({5,5});
  cout<<U<<endl;
  cout<<transp(U)*U<<endl;

  cout<<session<<endl;

  //Tensor<float> C(B);
  //cout<<C<<endl;

}

