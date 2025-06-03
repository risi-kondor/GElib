/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Etree.hpp"
#include "EtreeTensorNode.hpp"
#include "EtreeLoopNode.hpp"
#include "EinsumPrograms.hpp"


using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  EinsumForm form("ij,jk,kl->il");
  ContractionTrees ctrees(form);
  cout<<ctrees<<endl;
  
  Etree etree(ctrees.trees[0],{0,1,2,3});
  cout<<etree<<endl;


  //  EinsumPrograms programs(form);
  // C=A*B

  /*
  Etree etree;
  etree.root=to_share(new EtreeTensorNode(0,{0,1}));
  auto n=etree.insert_loop(etree.root,0);
  n=etree.insert_loop(n,2);
  n=etree.insert_contraction(n,1);

  EtreeParams params({3,3,3});
  cout<<etree.cpu_code(params)<<endl;
  */


}

