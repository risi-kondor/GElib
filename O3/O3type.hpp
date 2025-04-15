// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 20255, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _O3type
#define _O3type

#include "GElib_base.hpp"
#include "Gtype.hpp"
#include "O3group.hpp"
#include "O3index.hpp"


namespace GElib{

  class O3type: public Gtype<O3group>{
  public:

    typedef O3group GROUP;
    typedef O3index GINDEX;
    typedef Gtype<O3group> BASE;

    static constexpr O3index null_ix=O3index(-1);

    using BASE::parts;


    O3type(){}

    O3type(const std::map<O3index,int>& _parts):
      BASE(_parts){}

    O3type(const initializer_list<pair<O3index,int> >& list){
      for(auto p:list)
	parts[p.first]=p.second;
    }

    //SO3type(const vector<int>& v){
    //int l=0;
    //for(auto& p:v)
    //parts[l++]=p;
    //}


  public: // ---- I/O -------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::O3type";
    }

    string repr() const{
      ostringstream oss;
      oss<<"<O3type "<<str()<<">";
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

    friend ostream& operator<<(ostream& stream, const O3type& x){
      stream<<x.str(); return stream;
    }

  };



}

#endif 
