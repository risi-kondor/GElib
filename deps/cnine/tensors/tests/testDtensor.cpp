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
#include "Dtensor.hpp"
#include "Ltensor.hpp"

using namespace cnine;


int main(int argc, char** argv){
  cnine_session session;

  
  Dtensor A(dims(2,2),filltype=4,dtype=dint);
  Dtensor B(dims(2,2),filltype=4,dtype=dfloat);
  Dtensor C(dims(2,2),filltype=4,dtype=ddouble);
  Dtensor D(dims(2,2),filltype=4,dtype=dcfloat);

  cout<<A<<endl;
  cout<<B<<endl;
  cout<<C<<endl;
  cout<<D<<endl;

  cout<<"int:"<<endl;
  cout<<A.get_int(0,0)<<endl;
  cout<<B.get_int(0,0)<<endl;
  cout<<C.get_int(0,0)<<endl;
  cout<<D.get_int(0,0)<<endl;
  cout<<endl;

  cout<<"float:"<<endl;
  cout<<A.get_float(0,0)<<endl;
  cout<<B.get_float(0,0)<<endl;
  cout<<C.get_float(0,0)<<endl;
  cout<<D.get_float(0,0)<<endl;
  cout<<endl;

  cout<<"double:"<<endl;
  cout<<A.get_double(0,0)<<endl;
  cout<<B.get_double(0,0)<<endl;
  cout<<C.get_double(0,0)<<endl;
  cout<<D.get_double(0,0)<<endl;
  cout<<endl;

  cout<<"cfloat:"<<endl;
  cout<<A.get_cfloat(0,0)<<endl;
  cout<<B.get_cfloat(0,0)<<endl;
  cout<<C.get_cfloat(0,0)<<endl;
  cout<<D.get_cfloat(0,0)<<endl;
  cout<<endl;



}
