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
#include "RtensorObj.hpp"
#include "CnineSession.hpp"

using namespace cnine;

typedef RtensorObj rtensor; 


int main(int argc, char** argv){

  cnine_session session;

  cout<<endl;

  rtensor A=rtensor::sequential({4,4});
  cout<<"A="<<endl<<A<<endl<<endl;;

  rtensor B=rtensor::gaussian({4,4});
  cout<<"B="<<endl<<B<<endl<<endl;

  auto Av=A.view2();
  cout<<Av(2,3)<<endl;

}
