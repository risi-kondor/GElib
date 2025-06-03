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
typedef CtensorArray ctensora;


int main(int argc, char** argv){
  cnine_session genet;
  cout<<endl;

  ctensora A(dims(2),dims(2,2),[](const Gindex& aix, const Gindex& cix){return aix(0);});
  printl("A",A);

  ctensora B(dims(2),dims(2,2),[](const Gindex& aix, const Gindex& cix){return complex<float>(0,aix(0));});
  printl("B",B);

  CtensorA_add_plus_cop add_plus;

  ctensora Inner(dims(1,1),dims(2,2),fill::zero);
  InnerBiCmap(add_plus,Inner,A,B);
  printl("Inner",Inner);

  ctensora Outer(dims(2,2),dims(2,2),fill::zero);
  OuterBiCmap(add_plus,Outer,A,B);
  printl("Outer",Outer);


  cout<<"--------"<<endl;

  ctensora C(dims(2,2),dims(2,2),[](const Gindex& aix, const Gindex& cix){return aix(0);});
  ctensora D(dims(2,2),dims(2,2),[](const Gindex& aix, const Gindex& cix){return complex<float>(0,aix(1));});
  ctensor M(dims(2,2),[](const int i, const int j){return 3;});
  printl("C",C);
  printl("D",D);
  printl("M",M);

  ctensora Cwise(dims(2,2),dims(2,2),fill::zero);
  CellwiseBiCmap(add_plus,Cwise,C,D);
  printl("Cwise",Cwise);

  //ctensora Bcast(dims(2,2),dims(2,2),fill::zero);
  // BroadcastLeftBiCmap(add_plus,Bcast,ctensora(M),C);
  //printl("Bcast",Bcast);

  
  cout<<"--------"<<endl;

  ctensora E(dims(5,5),dims(2,2),[](const Gindex& aix, const Gindex& cix){
      return complex<float>(aix(0),aix(1));});
  ctensora F(dims(2,2),dims(2,2),[](const Gindex& aix, const Gindex& cix){
      return 0; 
      //return complex<float>(aix(0),aix(1));
    });
  printl("F",F);

  ctensora Conv0(dims(4,4),dims(2,2),fill::zero);
  Convolve2BiCmap(add_plus,Conv0,E,F);
  printl("Conv0",Conv0);


}

