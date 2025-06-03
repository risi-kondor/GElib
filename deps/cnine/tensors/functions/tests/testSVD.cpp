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
#ifdef _WITH_EIGEN
#include "SingularValueDecomposition.hpp"
#endif 

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  cout<<endl;


  Tensor<double> A=Tensor<double>::randn({5,5});
  cout<<A<<endl;


#ifdef _WITH_EIGEN
  auto svd=SingularValueDecomposition(A);
  //print(svd.S());
  //print(svd.U());
  //print(svd.V());

  auto U=svd.U();
  auto V=svd.V();
  auto S=svd.S();

  print(U*diag(S)*cnine::transp(V));
#endif
}
