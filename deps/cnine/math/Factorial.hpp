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

#ifndef _CnineFactorial
#define _CnineFactorial 

#include "Cnine_base.hpp"


namespace cnine{

  class Factorial{
  public:

    vector<int> fact;

    Factorial(){
      fact.push_back(1);
    }      

    int operator()(const int n){
      if(n<fact.size()) return fact[n]; 

      int m=fact.size();
      fact.resize(n+1);
      for(int i=m; i<=n; i++)
	fact[i]=fact[i-1]*i;

      return fact[n];
    }
  };

}

#endif
