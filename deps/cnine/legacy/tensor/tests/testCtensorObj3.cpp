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
#include "CnineSession.hpp"

using namespace cnine;

typedef CscalarObj cscalar; 
typedef CtensorObj ctensor; 


int main(int argc, char** argv){

  cnine_session session;

  cout<<endl;

  Gtensor<complex<float> > Gin(dims(4,4),fill::sequential);

  ctensor A(Gin);
  ctensor B(Gin);
  cout<<A<<endl;

  ctensor C=A*B;


  Gtensor<complex<float> > Gout=C.gtensor();
  cout<<Gout<<endl;
 
}
