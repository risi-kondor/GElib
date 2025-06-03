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
#include "CtensorB.hpp"
#include "CnineSession.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  CtensorB T=CtensorB::sequential({4,4,4});

  cout<<T<<endl; 

  //cout<<T.view3D()<<endl;

  cout<<T.view3D().slice0(0)<<endl;
  cout<<T.view3D().slice0(1)<<endl;
  cout<<T.view3D().slice0(2)<<endl;
  cout<<T.view3D().slice0(3)<<endl;

  cout<<T.view3D().slice1(0)<<endl;
  cout<<T.view3D().slice1(1)<<endl;
  cout<<T.view3D().slice1(2)<<endl;
  cout<<T.view3D().slice1(3)<<endl;

  cout<<T.view3D().slice2(0)<<endl;
  cout<<T.view3D().slice2(1)<<endl;
  cout<<T.view3D().slice2(2)<<endl;
  cout<<T.view3D().slice2(3)<<endl;



}
