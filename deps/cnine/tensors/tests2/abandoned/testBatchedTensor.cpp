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
#include "BatchedTensor.hpp"
#include "Tensor.hpp"
#include "CnineSession.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  cout<<endl;

  //BatchedTensor<float> A=BatchedTensor<float>::sequential(2,{3,3});
  BatchedTensor<float> A(3,{3,3},[&](const int b){return Tensor<float>::constant({3,3},b+2);});
  cout<<A<<endl;
  cout<<prod(A,A)<<endl;

  BatchedTensor<float> B(1,{3,3},[&](const int b){return Tensor<float>::constant({3,3},b+2);});
  cout<<prod(A,B)<<endl;

  Tensor<float> C=Tensor<float>::sequential({3,3});
  cout<<prod(A,C)<<endl;

}

