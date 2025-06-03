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
#include "Einsum2.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  
  //Ltensor<float> A(LtensorSpec<float>().batch(2).dims({2,2}).sequential());
  TensorView<float> x(dims(3,3,3),3,0);
  cout<<x<<endl;

  TensorView<float> y(dims(3,3,3),3,0);
  cout<<y<<endl;

  Einsum2 R("abc,abc->abc");
  cout<<R(x,y)<<endl;




  while(true){
    string str;
    getline(cin,str);
    cout<<Einsum2(str)(x,y)<<endl;
  }

}
