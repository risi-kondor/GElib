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


#ifndef __Llist
#define __Llist

#include "Cnine_base.hpp"


namespace cnine{


  template<typename TYPE>
  class Llist: public map<TYPE,int>{
  public:

    using map<TYPE,int>::end;
    using map<TYPE,int>::size;
    using map<TYPE,int>::insert;

    Llist(){}

    Llist(const initializer_list<TYPE>& list){
      int i=0;
      for(auto& p:list)
	insert({p,i++});
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int isize() const{
      return size();
    }

    void insert(const TYPE& ix){
      insert({ix,size()});
    }

    int operator()(const TYPE& ix) const{
      if(const_cast<Llist<TYPE>&>(*this).find(ix)==end()) return -1;
      return const_cast<Llist<TYPE>&>(*this).map<TYPE,int>::operator[](ix);
    }

    TYPE operator[](const int i) const{
      CNINE_ASSRT(i<size());
      for(auto& p:*this)
	if(p.second==i) return p.first;
      return TYPE();
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str() const{
      ostringstream oss;
      oss<<"{";
      int i=0;
      for(auto& p:*this){
	oss<<p.first;
	if(i++<size()-1) oss<<",";
      }
      oss<<"}";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Llist<TYPE>& x){
      stream<<x.str(); return stream;
    }

  };

}


#endif 


