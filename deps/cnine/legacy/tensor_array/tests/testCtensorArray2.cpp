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
#include "CtensorObj_funs.hpp"
#include "CtensorArray_funs.hpp"
#include "CtensorA_add_Mprod_cop.hpp"

#include "CnineSession.hpp"


using namespace cnine;

typedef CscalarObj cscalar;
typedef CtensorObj ctensor;
typedef CtensorArray ctensor_array;


int main(int argc, char** argv){

  cnine_session genet;

  cout<<endl;

  ctensor_array A=ctensor_array::sequential(dims(2,2),dims(4,4));
  cout<<"A="<<endl<<A<<endl<<endl;

  ctensor_array B=ctensor_array::sequential(dims(2,2),dims(4,4));
  cout<<"B="<<endl<<B<<endl<<endl;

  //A.map_binary(CtensorA_add_Mprod_cop(),A,B);
  //cout<<"A="<<endl<<A<<endl<<endl;

  //ctensor_array C=A*B;
  //cout<<"C="<<endl<<C<<endl<<endl;

  printl("A*B",A*B);

}
