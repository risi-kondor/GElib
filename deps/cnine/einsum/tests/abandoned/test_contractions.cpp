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
#include "EinsumN.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  //EinsumFormN f("ij,jk,kl->");
  //cout<<f<<endl;

  EinsumN esum("ij,jk,kl->");
  cout<<*esum.programs<<endl;
  esum.programs->latex();

  string latex_cmd("pdflatex temp");
  system(latex_cmd.c_str());

  string open_cmd("open temp.pdf");
  system(open_cmd.c_str());

}

