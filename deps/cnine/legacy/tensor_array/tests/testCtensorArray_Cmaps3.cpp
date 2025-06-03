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

#include "CtensorA_plus_cop.hpp"
#include "CtensorA_add_Mprod_cop.hpp"

#include "InnerCmap.hpp"
#include "OuterCmap.hpp"
#include "CellwiseBinaryCmap.hpp"
#include "MVprodCmap.hpp"
#include "BroadcastBinaryCmap.hpp"
#include "Convolve2Cmap.hpp"

#include "CnineSession.hpp"


using namespace cnine;

typedef CscalarObj cscalar;
typedef CtensorObj ctensor;

typedef CtensorA_plus_cop CtensorA_plus;


int main(int argc, char** argv){
  cnine_session genet;
  cout<<endl;

  CtensorArray A(dims(2),dims(2,2),-1,[](const Gindex& aix, const Gindex& cix){return aix(0);});
  CtensorArray B(dims(2),dims(2,2),-1,[](const Gindex& aix, const Gindex& cix){return complex<float>(0,aix(0));});

  printl("inner",inner<CtensorA_plus>(A,B));
  printl("outer",outer<CtensorA_plus>(A,B));

  cout<<"--------"<<endl;

  CtensorArray C(dims(2,2),dims(2,2),-1,[](const Gindex& aix, const Gindex& cix){return aix(0);});
  CtensorArray D(dims(2,2),dims(2,2),-1,[](const Gindex& aix, const Gindex& cix){
      return complex<float>(0,aix(1));});
  ctensor M(dims(2,2),[](const int i, const int j){return 3;});
  //printl("C",C);
  //printl("D",D);
  //printl("M",M);

  printl("cellwise",cellwise<CtensorA_plus>(C,D));
  printl("MVprod",MVprod<CtensorA_plus>(C,A));

  printl("broadcast",broadcast<CtensorA_plus>(M,C));
  
  cout<<"--------"<<endl;

  CtensorArray E(dims(5,5),dims(2,2),-1,[](const Gindex& aix, const Gindex& cix){
      return complex<float>(aix(0),aix(1));});
  CtensorArray F(dims(2,2),dims(2,2),-1,[](const Gindex& aix, const Gindex& cix){
      return 0;});

  printl("convolve2",convolve2<CtensorA_plus>(E,F));

  cout<<"--------"<<endl;

  auto Ag=A.to(deviceid::GPU0);
  auto Bg=B.to(deviceid::GPU0);
  auto Cg=C.to(deviceid::GPU0);
  auto Dg=D.to(deviceid::GPU0);
  auto Eg=E.to(deviceid::GPU0);
  auto Fg=F.to(deviceid::GPU0);
  auto Mg=M.to(deviceid::GPU0);

  cout<<"--------"<<endl;
  printl("inner",inner<CtensorA_plus>(Ag,Bg));
  printl("outer",outer<CtensorA_plus>(Ag,Bg));
  printl("cellwise",cellwise<CtensorA_plus>(Cg,Dg));
  printl("MVprod",MVprod<CtensorA_plus>(Cg,Ag));
  printl("broadcast",broadcast<CtensorA_plus>(Mg,Cg));
  printl("convolve2",convolve2<CtensorA_plus>(Eg,Fg));
  exit(0);

}

