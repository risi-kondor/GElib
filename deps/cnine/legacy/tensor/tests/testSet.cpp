/*
 * This file is part of cnine, a lightweight C++ tensor library. 
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
//#include "CtensorB.hpp"
#include "CnineSession.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;


  set<int> A={1,3,2};
  set<int> B={2,1,2,3,4};

  for(auto p: A)
    cout<<p<<endl; 

  for(auto p: B)
    cout<<p<<endl; 

  cout<<(A==B)<<endl;

  cout<<*A.rbegin()<<endl;

}
