// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _SO3type
#define _SO3type

#include "GElib_base.hpp"
#include "SO3group.hpp"
#include "Gtype.hpp"


namespace GElib{

  //class SO3type: public Gtype<int>{
  class SO3type: public Gtype<SO3group>{
  public:

    typedef Gtype<SO3group> BASE;
    typedef int IRREP_IX;
    typedef SO3group Group;

    static constexpr int null_ix=-1;

    using BASE::parts;


    SO3type(){}

    SO3type(const std::map<int,int>& _parts):
      BASE(_parts){}

    SO3type(const initializer_list<pair<int,int> >& list){
      for(auto p:list)
	parts[p.first]=p.second;
    }

    SO3type(const vector<int>& v){
      int l=0;
      for(auto& p:v)
	parts[l++]=p;
    }


  public: // ---- I/O -------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::SO3type";
    }

    string repr() const{
      ostringstream oss;
      oss<<"<SO3type "<<str()<<">";
      return oss.str();
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<"(";
      for(auto& p:parts)
	oss<<p.first<<":"<<p.second<<",";
      oss<<"\b)";
      return oss.str();
    }
    
    string to_print(const string indent="") const{
      return repr();
    }

    friend ostream& operator<<(ostream& stream, const SO3type& x){
      stream<<x.str(); return stream;
    }

  };



}

#endif 
