/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2022, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _FFactorial
#define _FFactorial

#include "Cnine_base.hpp"
#include <map>
#include "frational.hpp"


namespace cnine{

  class FFactorial{
  public:

    vector<frational> f;

    FFactorial(){
      f.push_back(1);
    }

    frational operator()(const int x){
      extend(x);
      return f[x];
    }
    
    void extend(const int x){
      if(x<f.size()) return;
      for(int i=f.size(); i<=x; i++){
	frational I(i);
	f.push_back(f.back()*I);
      }
    }

  };

}

#endif 
