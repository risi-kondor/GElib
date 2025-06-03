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

  ctensor T=ctensor::sequential(dims(4,4));


  ctensora A=ctensora::zero(dims(2,2),dims(4,4));
  printl("A",A);

  ctensora B=ctensora::sequential(dims(2,2),dims(4,4));
  printl("B",B);

  ctensora C=ctensora::gaussian(dims(2,2),dims(4,4));
  printl("C",C);

  
  //ctensora T1(T);
  //printl("T1",T1);

  //ctensora T2(dims(2,2),T);
  //printl("T2",T2);


  ctensora L1(dims(2,2),dims(4,4),[](const Gindex& aix, const Gindex& ix){
      return complex<float>(aix(dims(2,2)),ix(dims(4,4)));});
  printl("L1",L1);
    

}
