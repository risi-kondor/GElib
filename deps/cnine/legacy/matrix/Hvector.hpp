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


#ifndef _CnineHvector
#define _CnineHvector

#include "Cnine_base.hpp"
#include <unordered_map>



namespace cnine{

  // deprecated
  template<typename TYPE>
  class Hvector: public unordered_map<int, TYPE>{
  public:

    int n=0;

    ~Hvector(){}


  public:

    Hvector(const int _n): n(_n){}

    
  public: // ---- Boolean ----------------------------------------------------------------------------------


    bool operator==(const Hvector& x){
      for(auto p: *this){
	auto it=x.find(p.first);
	if(it==x.end()) return false;
	if(p.second!=it->second) return false;
      }
      return true;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    TYPE operator()(const int i){
      return (*this)[i];
    }

    void set(const int i, const TYPE& v){
      (*this)[i]=v;
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void forall_nonzero(const std::function<void(const int, const TYPE& )>& lambda) const{
      for(auto p:*this)
	lambda(p.first,p.second);
    }



  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for(auto p:*this){
	oss<<"("<<p.first<<","<<p.second<<")";
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Hvector& x){
      stream<<x.str(); return stream;}

  };

}


namespace std{
  /*
  template<>
  struct hash<cnine::Hvector>{
  public:
    size_t operator()(const cnine::Hvector<TYPE>& x) const{
      size_t t=1;
      for(auto p: x){
	t=(t<<1)^hash<int>()(p.first);
	t=(t<<1)^hash<int>()(p.second);
      }
      return t;
    }
  };
  */
}


#endif 
