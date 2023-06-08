
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3BiType
#define _SO3BiType

#include "GElib_base.hpp"
#include <unordered_map>
#include "SO3type.hpp"


namespace GElib{

  class SO3BiType: public std::unordered_map<pair<int,int>,int>{
  public:

    SO3BiType(){};

    SO3BiType(const initializer_list<std::pair<std::pair<int,int>,int> >& list){
      for(auto p:list)
	(*this)[p.first]=p.second;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int operator()(const int l1, const int l2) const{
      auto it=find(std::make_pair(l1,l2));
      if(it!=end()) return it->second;
      return 0;
    }

    void set(const int l1, const int l2, const int m){
      (*this)[make_pair(l1,l2)]=m;
    }

    void inc(const int l1, const int l2, const int m){
      (*this)[make_pair(l1,l2)]+=m;
    }

    void for_each(const std::function<void(const int l1, const int l2, const int m)>& lambda) const{
      for(auto& p:*this)
	lambda(p.first.first,p.first.second,p.second);
    }


  public:

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      oss<<"[";
      for(auto& p: *this)
	oss<<"(("<<p.first.first<<","<<p.first.second<<"):"<<p.second<<")";
      oss<<"]";
      return oss.str();
    }

    string repr(const string indent="") const{
      return indent+"<GElib::SO3BiType"+str()+">";
    }

    friend ostream& operator<<(ostream& stream, const GElib::SO3BiType& x){
      stream<<x.str(); return stream;}

  };


  // ---- Post-class functions -------------------------------------------------------------------------------




}

/*
namespace std{
  template<>
  struct hash<GElib::SO3BiType>{
  public:
    size_t operator()(const GElib::SO3BiType& tau) const{
      size_t h=0;
      for(auto p:tau)
      h=(h<<1)^hash<int>()(p);
      return h;
    }
  };
}
*/



#endif

