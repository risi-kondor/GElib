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

  ctensor A({4,4},fill::zero);
  ctensor B({4},fill::sequential);
  ctensor C({2,4},fill::sequential);

  A.add_to_chunk(0,1,C);
  
  cout<<"A="<<endl<<A<<endl<<endl;;

  A.add_to_slice(0,3,B);
  cout<<"A="<<endl<<A<<endl;

  C.add_chunk_of(A,0,1,2);
  cout<<C<<endl;

  printl("A.slice(0,2)",A.slice(0,2));

  printl("A.chunk(0,1,2)",A.chunk(0,1,2));

  cout<<"-----------"<<endl<<endl;

  ctensor D({4,4},fill::zero);
  vector<const ctensor*> v;
  for(int i=0; i<4; i++) v.push_back(&B);
  //D.add_to_slices(1,v);
  D.add_to_slices(1,B,B,B,B);
  printl("D",D);

  cout<<"-----------"<<endl<<endl;

  //ctensor E(1,B,B,B,B);
  //ctensor E=stack(1,B,B,B,B);
  //printl("E",E);

  //ctensor F(fill::cat,0,B,B);
  //ctensor F=cat(0,B,B);
  //printl("F",F);


}
