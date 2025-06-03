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
#include "TensorPack.hpp"
#include "Tensor.hpp"
#include "CnineSession.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  cout<<endl;

  Gdims dims({3,3});

  TensorPack<float> A=TensorPack<float>::sequential(3,dims);
  cout<<A<<endl;

  Tensor<float> B0=Tensor<float>::sequential({2,2});
  Tensor<float> B1=Tensor<float>::sequential({2,3});
  Tensor<float> B2=Tensor<float>::sequential({4,4});
  
  TensorPack<float> B({B0,B1,B2});
  cout<<B<<endl;
    

}
