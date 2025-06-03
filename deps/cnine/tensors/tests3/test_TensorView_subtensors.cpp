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

  auto A=TensorView<float>(cdims=Gdims({5,5}),filltype=3,device=0); 
  show(A);

  show(A.row(1));
  show(A.rows(1,2));
  show(A.col(2));
  show(A.cols(2,2));

  show(A.block(1,1,2,2));

  show(A.diag());

  show(A.transp());
  show(A.reshape({25}));

  auto B=TensorView<float>(cdims=Gdims({5,5,5}),filltype=3,device=0); 
  show(B);
  
  show(B.slice(0,1));
  show(B.diag({0,1,2}));

  show(B.transp(1,2));
}
