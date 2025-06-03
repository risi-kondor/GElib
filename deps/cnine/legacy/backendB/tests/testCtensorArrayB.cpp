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
#include "CnineSession.hpp"
#include "CtensorArrayB.hpp"
#include "RtensorA.hpp"

using namespace cnine;


int main(int argc, char** argv){
  cnine_session session;
  cout<<endl;

  int n=4;
  Gdims adims({n});
  Gdims cdims({2,2});

  CtensorArrayB u=CtensorArrayB::gaussian(adims,cdims);
  PRINTL(u);

  RtensorA M=RtensorA::zero({n,n});
  M.set(0,1,1.0);
  M.set(0,3,1.0);
  M.set(2,2,1.0);
  cout<<M<<endl;

  Rmask1 mask=Rmask1::matrix(M.view2());
  cout<<mask<<endl;
  cout<<mask.inv()<<endl;
  
  auto w=u.gather(mask);
  PRINTL(w);
}


