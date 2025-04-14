// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _GElibSO3vec
#define _GElibSO3vec

#include "SO3part.hpp"
#include "SO3type.hpp"
#include "Gvec.hpp"


namespace GElib{


  template<typename TYPE>
  class SO3vec: public Gvec<SO3vec<TYPE>,SO3part<TYPE> >{
  public:

    typedef Gvec<SO3vec,SO3part<TYPE> > BASE;

    typedef SO3group Group;
    typedef int IrrepIx; 
    typedef SO3type GTYPE; 

    using BASE::BASE;
    using BASE::parts;
    using BASE::unroller;
    using BASE::get_dev;
    using BASE::get_tau;
    using BASE::str;


  public: // ---- Named parameter constructors ---------------------------------------------------------------


    template<typename ARG0, typename... Args, 
	     typename = typename std::enable_if<
    std::is_same<TtypeArgument, ARG0>::value ||
    std::is_same<cnine::BatchArgument, ARG0>::value ||
    std::is_same<cnine::GridArgument, ARG0>::value ||
    std::is_same<cnine::FillArgument, ARG0>::value ||
    std::is_same<cnine::DeviceArgument, ARG0>::value, ARG0>::type>
    SO3vec(const ARG0& arg0, const Args&... args){
      typename BASE::vparams v;
      unroller(v,arg0,args...);
      BASE::reset(v);
      if(v.tau.has_value()==false) 
	throw std::invalid_argument("GElib error: constructor of SO3vec must have an ttype argument.");
      SO3type tau=any_cast<SO3type>(v.tau);
      for(auto& p:tau.parts){
	if(v.gdims.size()>0) parts.emplace(p.first,SO3part<TYPE>(v.b,v.gdims,p.first,p.second,v.fcode,v.dev));
	else parts.emplace(p.first,SO3part<TYPE>(v.b,p.first,p.second,v.fcode,v.dev));
      }
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    //SO3type get_tau() const{
    //SO3type R;
    //for(auto& p:parts)
    //R[p.first]=p.second.getn();
    //return R;
    //}


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
    
    friend ostream& operator<<(ostream& stream, const SO3vec& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 
