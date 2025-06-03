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

#ifndef _Primes
#define _Primes

#include "Cnine_base.hpp"


namespace cnine{

  class Primes: public vector<int>{
  public:

    int limit=1;

    void extend(const int lim){
      if(lim<=limit) return;
      for(int i=limit+1; i<=lim; i++){
	int x=i;
	for(auto p: *this){
	  while(x%p==0) {x/=p;}
	  if(x==1) break;
	}
	if(x>1) push_back(x);
      }
      limit=lim;
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"(";
      for(int i=0; i<size(); i++){
	oss<<(*this)[i];
	if(i<size()-1) oss<<",";
      }
      oss<<")";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Primes& x){
      stream<<x.str(); return stream;}

  };

}

#endif 
