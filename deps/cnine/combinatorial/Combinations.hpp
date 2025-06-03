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

#ifndef _Combinations
#define _Combinations

#include "Cnine_base.hpp"
#include "CombinationsBank.hpp"

extern cnine::CombinationsBank combinations_bank;


namespace cnine{


  class Combinations{
  public:

    CombinationsB* obj;

    Combinations(const int n, const int m){
      obj=combinations_bank.get(n,m);
    }


  public:

    int getN() const{
      return obj->getN();
    }

    Combination operator[](const int i) const{
      return (*obj)[i];
    }

    int index(const Combination& v) const{
      return obj->index(v);
    }

    void for_each(std::function<void(Combination) > fn) const{
      return obj->for_each(fn);
    }


  public:

    string str(const string indent="") const{
      ostringstream oss;
      for_each([&](const Combination& v){
	  oss<<v.str()<<endl;
	});
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Combinations& x){
      stream<<x.str(); return stream;
    }


  };


}

#endif 
