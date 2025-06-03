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
#include "RtensorObj_funs.hpp"
#include "RtensorArray_funs.hpp"

#include "CnineSession.hpp"


using namespace cnine;

typedef RscalarObj rscalar;
typedef RtensorObj rtensor;
typedef RtensorArray rtensor_array;


int main(int argc, char** argv){

  cnine_session genet;

  cout<<endl;

  rtensor_array A=rtensor_array::sequential(dims(2,2),dims(4,4));
  cout<<"A="<<endl<<A<<endl<<endl;

  cout<<"A.reduce(0)="<<endl<<A.reduce(0)<<endl;
  cout<<"A.reduce(1)="<<endl<<A.reduce(1)<<endl;

  rtensor_array B=rtensor_array::sequential(dims(1,1),dims(4,4));

  cout<<"B.widen(0,3)="<<endl<<B.widen(0,3)<<endl;
  cout<<"B.widen(1,3)="<<endl<<B.widen(1,3)<<endl;
  cout<<"B.widen(2,3)="<<endl<<B.widen(2,3)<<endl;

  rtensor_array C(dims(2,2),dims(4,4),[](const Gindex& aix, const Gindex& ix){
      return aix(dims(2,2))+ix(dims(4,4));});
  cout<<"C="<<endl<<C<<endl;
  
  rtensor M1=rtensor::sequential(dims(3,3));
  //rtensor_array D=broadcast(dims(2,2),M1);
  //cout<<"D="<<endl<<D<<endl; 

  //rtensor M2=rtensor::sequential(dims(2,2));
  //D*=scatter(M2);
  //cout<<"D="<<endl<<D<<endl; 

  //rtensor_array E=D*(broadcast(M1));
  //cout<<"E="<<endl<<E<<endl; 

  //rtensor_array F=rtensor_array::identity(dims(2,2),dims(4,4));
  //printl("F",F);

}
