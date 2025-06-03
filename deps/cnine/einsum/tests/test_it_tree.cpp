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
#include "ContractionTrees.hpp"
#include "it_tree.hpp"


using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;
  cout<<endl;

  //index_set rix({0,2});
  //it_tree itt(rix);

  EinsumForm form("ij,jk,kl->il");
  ContractionTrees ctrees(form);
  cout<<ctrees<<endl;

  it_tree itt(ctrees.trees[0]);
  //itt.find_insertion_point(itt.root,index_set({0,1,2,3}));

  cout<<itt.code()<<endl;

}

