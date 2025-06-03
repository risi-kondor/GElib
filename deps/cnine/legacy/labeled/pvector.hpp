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


#ifndef __pvector
#define __pvector

#include "Cnine_base.hpp"

namespace cnine{

  template<typename TYPE>
  class pvector: public vector<TYPE*>{
  public:

    using vector<TYPE*>::size;
    using vector<TYPE*>::operator[];
    using vector<TYPE*>::push_back;

    pvector(){}

    pvector(const int n):
      vector<TYPE*>(n){}


    ~pvector(){
      for(auto p:*this)
	delete p;
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    pvector(const pvector& x){
      for(auto p:x)
	push_back(p->clone());
    }

    pvector(pvector&& x){
      for(auto p:x)
	push_back(p);
      x.clear();
    }

    pvector& operator=(const pvector& x){
      for(auto p:*this)
	delete p;
      for(auto p:x)
	push_back(p->clone());
    }

    pvector& operator=(pvector&& x){
      for(auto p:*this)
	delete p;
      for(auto p:x)
	push_back(p);
      x.clear();
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    pvector remove(const int j) const{
      CNINE_ASSRT(j<size());
      pvector<TYPE> r(size()-1);
      for(int i=0; i<j; i++)
	r[i]=(*this)[i]->clone();
      for(int i=j+1; i<size(); i++)
	r[i-1]=(*this)[i]->clone();
      return r;
    }


  };

}

#endif 
