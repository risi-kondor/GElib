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


using namespace cnine;

int main(int argc, char** argv){
  cnine_session session(4);

  frational x(26);
  cout<<x<<endl;

  frational y(28,6);
  cout<<y<<endl;

}
