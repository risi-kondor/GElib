// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _SO3vecE
#define _SO3vecE

#include "Gvec.hpp"
#include "SO3part.hpp"
#include "SO3type.hpp"


namespace GElib{


  template<typename TYPE>
  class SO3vec: public Gvec<SO3part<TYPE> >{
  public:

    typedef Gvec<SO3part<TYPE> > BASE;

    using BASE::parts;
    using BASE::unroller;
    using BASE::get_dev;


  public: // ---- Named parameter constructors ---------------------------------------------------------------


    template<typename... Args>
    SO3vec(const Args&... args){
      typename BASE::vparams v;
      unroller(v,args...);
      BASE::reset(v);
      SO3type tau=any_cast<SO3type>(v.tau);
      for(auto& p:tau.map)
	if(v.gdims.size()>0) parts.emplace(p.first,SO3part<TYPE>(v.b,v.gdims,p.first,p.second,v.fcode,v.dev));
	else parts.emplace(p.first,SO3part<TYPE>(v.b,p.first,p.second,v.fcode,v.dev));
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    SO3type get_tau() const{
      SO3type R;
      for(auto& p:parts)
	R[p.first]=p.second.get_nc();
      return R;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::SO3vec";
    }

    string repr() const{
      ostringstream oss;
      oss<<"<SO3vec";
      if(BASE::is_batched() && BASE::nbatch()>1) oss<<" b="<<BASE::nbatch();
      if(BASE::is_grid()) oss<<" grid="<<BASE::gdims();
      oss<<" tau="<<get_tau();
      if(get_dev()>0) oss<<" device="<<BASE::get_dev();
      oss<<">";
      return oss.str();
    }
    
    string str(const string indent="") const{
      ostringstream oss;
      for(auto& p:parts)
	oss<<p.second<<endl;
      return oss.str();
    }

    string to_print(const string indent="") const{
      ostringstream oss;
      oss<<indent<<repr()<<":"<<endl;
      oss<<str(indent+"  ")<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const SO3vec& x){
      stream<<x.str(); return stream;
    }




  };

}

#endif 
