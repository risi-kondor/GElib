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
#include "RscalarObj_funs.hpp"

using namespace cnine;

typedef RscalarObj rscalar;


int main(int argc, char** argv){

  rscalar A=2.0;
  rscalar B=3.0;
  rscalar C=-1.0;

  cout<<endl; 
  cout<<" A+B = "<<A+B<<endl;
  cout<<" A-B = "<<A-B<<endl;
  cout<<" A*B = "<<A*B<<endl;
  cout<<" A/B = "<<A/B<<endl;
  cout<<endl; 

  cout<<" abs(A) = "<<abs(A)<<endl;
  cout<<" inp(A,B) = "<<inp(A,B)<<endl;
  cout<<" ReLU(C) = "<<ReLU(C,0.1)<<endl;

  cout<<" fn(A) = "<<A.apply([](const float x){return x*x+3;})<<endl; 

  cout<<endl; 

}
