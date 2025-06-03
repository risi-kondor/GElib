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


#ifndef __GindexSet
#define __GindexSet

#include "Cnine_base.hpp"


namespace cnine{


  class GindexSet: public set<int>{
  public:

    using set::set;


  public:

    int first() const{
      return *this->begin();
    }

    int last() const{
      return *this->rbegin();
    }

    bool is_contiguous() const{
      int i=first()-1;
      for(auto p:*this)
	if(p!=(i++)) return false;
      return true;
    }

    bool is_disjoint(const GindexSet& y) const{
      for(auto p:*this)
	if(y.find(p)!=y.end()) return false;
      return true;
    }

    bool covers(const int n) const{
      for(int i=0; i<n; i++)
	if(this->find(i)==this->end()) return false;
      return true;
    }

    bool covers(const int n, const GindexSet& x) const{
      for(int i=0; i<n; i++)
	if((this->find(i)==this->end())&&(x.find(i)==x.end())) return false;
      return true;
    }

    bool covers(const int n, const GindexSet& x, const GindexSet& y) const{
      for(int i=0; i<n; i++)
	if((this->find(i)==this->end())&&(x.find(i)==x.end())&&(x.find(i)==x.end())) return false;
      return true;
    }

    int back() const{
      return *(this->rbegin());
    }


  };

}


#endif 
