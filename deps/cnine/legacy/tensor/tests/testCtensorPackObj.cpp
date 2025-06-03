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
#include "CtensorPackObj_funs.hpp"
#include "CnineSession.hpp"

using namespace cnine;

typedef CscalarObj cscalar; 
typedef CtensorObj ctensor; 
typedef CtensorPackObj ctensorpack; 



int main(int argc, char** argv){

  cnine_session session;

  cout<<endl;

  ctensorpack A=ctensorpack::sequential(3,dims(4,4));
  cout<<"A="<<endl<<A<<endl<<endl;;

  ctensorpack B=ctensorpack::gaussian(3,{4,4});
  cout<<"B="<<endl<<B<<endl<<endl;

  cscalar c(2.0);

  cout<<"A+B="<<endl<<A+B<<endl<<endl;
  cout<<"A-B="<<endl<<A-B<<endl<<endl;
  cout<<"c*A="<<endl<<c*A<<endl<<endl;
  cout<<"A*B="<<endl<<A*B<<endl<<endl;
  cout<<endl; 

  cout<<"norm2(A)="<<endl<<norm2(A)<<endl<<endl;
  cout<<"inp(A,B)="<<endl<<inp(A,B)<<endl<<endl;
  //cout<<"ReLU(A,0.1)="<<endl<<ReLU(A,0.1)<<endl<<endl;
  cout<<endl; 
  

  ctensorpack C=ctensorpack::zero(3,{4,4});
  C+=A;
  C+=A;
  C+=A;
  C+=A;
  cout<<"A+A+A+A="<<endl<<C<<endl;
  cout<<endl; 

  //CscalarObj x=A(1,1);
  //cout<<"A(1,1) = "<<x<<endl<<endl; 

  //CscalarObj y(77);
  //A.set(1,1,y);
  //cout<<"A="<<endl<<A<<endl<<endl;  

  cout<<endl;
}
