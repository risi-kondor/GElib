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
#include "LtensorApack.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  std::map<int,Gdims> ddims({{1,{2,2}},{5,{3,3}}});
  
  LtensorApack<int,float> B=LtensorApack<int,float>::gaussian().batch(2).dims(ddims);
  cout<<B<<endl;

  auto C=LtensorApack<int,float>::zero().batch(2).dims(ddims)();
  cout<<C<<endl;
  

}
