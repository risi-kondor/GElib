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

  ctensora A=ctensora::sequential(dims(2,2),dims(4,4));
  printl("A",A);

  cout<<"A.reduce(0)="<<endl<<A.reduce(0)<<endl;
  cout<<"A.reduce(1)="<<endl<<A.reduce(1)<<endl;

  ctensora B=ctensora::sequential(dims(1,1),dims(4,4));

  cout<<"B.widen(0,3)="<<endl<<B.widen(0,3)<<endl;
  cout<<"B.widen(1,3)="<<endl<<B.widen(1,3)<<endl;
  cout<<"B.widen(2,3)="<<endl<<B.widen(2,3)<<endl;

  ctensora C(dims(2,2),dims(4,4),[](const Gindex& aix, const Gindex& ix){
		   return complex<float>(aix(dims(2,2)),ix(dims(4,4)));});
  cout<<"C="<<endl<<C<<endl;
  
  //ctensor M1=ctensor::sequential(dims(3,3));
  //ctensora D=broadcast(dims(2,2),M1);
  //cout<<"D="<<endl<<D<<endl; 

  //ctensor M2=ctensor::sequential(dims(2,2));
  //D*=scatter(M2);
  //cout<<"D="<<endl<<D<<endl; 

  //ctensora E=D*(broadcast(M1));
  //cout<<"E="<<endl<<E<<endl; 

  //ctensora F=ctensora::identity(dims(2,2),dims(4,4));
  //printl("F",F);

}
