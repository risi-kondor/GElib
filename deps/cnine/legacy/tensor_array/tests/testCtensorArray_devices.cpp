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
#include "CtensorArray.hpp"

#include "CnineSession.hpp"


using namespace cnine;

typedef CscalarObj cscalar;
typedef CtensorObj ctensor;
typedef CtensorArray ctensora;


int main(int argc, char** argv){

  cnine_session session;
  cout<<endl;

  ctensora A=ctensora::zero(dims(2,2),dims(4,4),deviceid::GPU0);
  printl("A",A);

  ctensora B=ctensora::sequential(dims(2,2),dims(4,4),deviceid::GPU0);
  printl("B",B);


  ctensora T=ctensora::sequential(dims(2,2),dims(4,4));
  ctensora T1=T.to(deviceid::GPU0);
  printl("T1",T1);

  ctensora T2=T1.to(deviceid::CPU);
  printl("T2",T2);

}
