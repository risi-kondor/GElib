/* This file is part of cnine, a lightweight C++ tensor library. 
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
#include "Ptens_base.cpp"
#include "MultiLoop.hpp"
#include "Ltensor.hpp"
#include "SimpleMemoryManager.hpp"


using namespace cnine;


int main(int argc, char** argv){

  cnine_session session(4);
  //SimpleMemoryManager mm(4096);
  SimpleMemoryManager mm(Mbytes(2));
  cout<<mm<<endl;

  Ltensor<float> A(mm,{2,2},0);
  Ltensor<float>* B=new Ltensor<float>(mm,{2,2},0);
  Ltensor<float> C(mm,{2,2},0);
  cout<<mm<<endl;

  delete B;
  cout<<mm<<endl;


}

