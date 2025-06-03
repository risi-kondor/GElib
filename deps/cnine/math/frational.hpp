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

#ifndef _frational
#define _frational

#include "Cnine_base.hpp"
#include "Primes.hpp"

namespace cnine{

  extern Primes primes;


  class frational{
  public:

    map<int,int> factors;

    frational(){}

    frational(int x){
      primes.extend(x);
      for(auto p:primes){
	int i=0;
	while(x%p==0) {x/=p; i++;}
	factors[p]+=i;
	if(x==1) break;
      }
    }

    frational(int p, int q){
      frational P(p);
      frational Q(q);
      (*this)=P/Q;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    operator double() const{
      double r=1.0;
      for(auto& x:factors)
	r*=pow(x.first,x.second);
      return r;
    }

    double log() const{
      double r=0;
      for(auto& x:factors)
	r+=::log(x.first)*x.second;
      return r;
    }

    bool p_not_one() const{
      for(auto& x:factors)
	if(x.second>0) return true;
      return false;
    }

    bool q_not_one() const{
      for(auto& x:factors)
	if(x.second<0) return true;
      return false;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    frational operator/(const frational& q){
      frational r(*this);
      for(auto& x:q.factors)
	r.factors[x.first]-=x.second;
      return r;
    }

    frational operator*(const frational& y){
      frational r(*this);
      for(auto& x:y.factors)
	r.factors[x.first]+=x.second;
      return r;
    }

    frational operator*(const int y){
      return (*this)*frational(y);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str() const{
      ostringstream oss;

      if(p_not_one()){
	oss<<"(";
	int i=0;
	for(auto& x: factors){
	  if(x.second>0){
	    if(i++>0) oss<<"*";
	    if(x.second==1) oss<<x.first; 
	    else oss<<x.first<<"^"<<x.second;
	  }
	}
	oss<<")";
      }else{
	oss<<1;
      }

      if(q_not_one()){
	oss<<"/(";
	int i=0;
	for(auto& x: factors){
	  if(x.second<0){
	    if(i++>0) oss<<"*";
	    if(x.second==-1) oss<<x.first; 
	    else oss<<x.first<<"^"<<-x.second;
	  }
	}
	oss<<")";
      }
     
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const frational& x){
      stream<<x.str(); return stream;}

  };

}

#endif 
