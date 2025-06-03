/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#include "Cnine_base.cpp"
#include "TensorView.hpp"
#include "CnineSession.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;
  cout<<endl;

  show(TensorView<float>({3,3},3,0)); // basic universal constructor 

  show(TensorView<float>::zero({3,3})); 
  show(TensorView<float>::raw({3,3})); 
  show(TensorView<float>::ones({3,3})); 
  show(TensorView<float>::sequential({3,3})); 
  show(TensorView<float>::gaussian({3,3})); 
  show(TensorView<float>::identity({3,3})); 
  show(TensorView<float>::unit(5,1)); 
  show(TensorView<float>::random_unitary({3,3})); 

  show(TensorView<float>::vec({3,5,9}));
  show(TensorView<float>({{3,5,9},{1,1,2}}));

  show(TensorView<float>(cdims=Gdims({4,4}),filltype=3,device=0)); 

  /*
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
  */

}

