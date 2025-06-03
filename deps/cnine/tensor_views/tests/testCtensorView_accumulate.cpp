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
#include "CtensorB.hpp"
#include "RtensorObj.hpp"
#include "Rmask1.hpp"
#include "AccumulateCmap.hpp"
#include "Ctensor1view_add.hpp"
//#include "TensorView_accumulator.hpp"
#include "Aggregator.hpp"

#include "CnineSession.hpp"


using namespace cnine;

typedef RtensorObj rtensor;
typedef CtensorB ctensor;


int main(int argc, char** argv){

  cnine_session genet;

  int n=5;
  device dev=deviceid::GPU0;
  cout<<endl;

  ctensor A=ctensor::zero({n,1});
  ctensor B=ctensor::sequential({n,1});

  rtensor M=rtensor::zero({n,n});
  M.set(0,1,1.0);
  M.set(0,3,1.0);
  M.set(2,2,1.0);
  cout<<M<<endl;

  Rmask1 mask=Rmask1::matrix(M.view2());
  cout<<mask<<endl;
  cout<<mask.inv()<<endl;

  //Ctensor1view_add op;
  //AccumulateCmap(op,A.view2(),B.view2(),mask);
  //print(A);

  //Aggregator(A.view2(),B.view2(),mask);
  //Aggregator(A,B,mask);
  A.add_gather(B,mask);
  print(A);

#ifdef _WITH_CUDA 

  ctensor Ag=ctensor::zero({n,1},1);
  ctensor Bg=B.to(dev);

  //Aggregator(Ag.view2(),Bg.view2(),mask);
  //Aggregator(Ag,Bg,mask);
  Ag.add_gather(Bg,mask);
  print(Ag);

#endif 

  
  ctensor C=ctensor::zero({n,3,3});
  ctensor D=ctensor::sequential({n,3,3});

  //Aggregator(C.view3(),D.view3(),mask);
  //Aggregator(C,D,mask);
  C.add_gather(D,mask);
  print(C);


}

