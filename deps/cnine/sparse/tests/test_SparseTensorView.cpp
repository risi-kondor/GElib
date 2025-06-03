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
#include "CnineSession.hpp"
#include "SparseTensorView.hpp"
#include "GatherMapB.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  TensorView<int> list({{1,2},{2,2},{3,0}});
  SparseTensorView<float> A({3,3,3,3},0,list,3);

  show(A);
}
