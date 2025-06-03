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
#include "LtensorPack.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  
  LtensorPack<float> B=LtensorPack<float>::gaussian().batch(2).dims({{1,1},{2,2},{3,3}});
  cout<<B<<endl;

  auto C=LtensorPack<float>::zero().batch(2).dims({{2,2}})();
  cout<<C<<endl;
  

}
