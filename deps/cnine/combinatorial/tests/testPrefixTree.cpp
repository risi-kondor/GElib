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

#include "CnineSession.hpp"
#include "PrefixTree.hpp"

using namespace cnine;

int main(int argc, char** argv){
  cnine_session session(4);

  PrefixTree<int> T; 

  T.add_path({1,2,3});
  T.add_path({1,2,4});
  T.add_path({99});

  cout<<T<<endl;


}
