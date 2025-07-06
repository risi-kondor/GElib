/*
 * This file is part of GElib, a C++/CUDA library for group equivariant 
 * tensor operations. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in orginal
 * or modified form) must retain this copyright notice and must be 
 * accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _GElibO3vec
#define _GElibO3vec

#include "O3part.hpp"
#include "O3type.hpp"
#include "Gvec.hpp"


namespace GElib{


  template<typename TYPE>
  class O3vec: public Gvec<O3vec<TYPE>,O3part<TYPE> >, public GvecBase{
  public:

    typedef Gvec<O3vec,O3part<TYPE> > BASE;

    typedef O3group Group;
    typedef int IrrepIx; 
    typedef O3type GTYPE; 

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
    O3vec(const ARG0& arg0, const Args&... args){
      typename BASE::vparams v;
      unroller(v,arg0,args...);
      BASE::reset(v);
      if(v.tau.has_value()==false) 
	throw std::invalid_argument("GElib error: constructor of O3vec must have an ttype argument.");
      O3type tau=any_cast<O3type>(v.tau);
      for(auto& p:tau.parts){
	if(v.gdims.size()>0) parts.emplace(p.first,O3part<TYPE>(p.first,v.b,v.gdims,p.second,v.fcode,v.dev));
	else parts.emplace(p.first,O3part<TYPE>(p.first,v.b,p.second,v.fcode,v.dev));
      }
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    //O3type get_tau() const{
    //O3type R;
    //for(auto& p:parts)
    //R[p.first]=p.second.getn();
    //return R;
    //}


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::O3vec";
    }

    string repr() const{
      ostringstream oss;
      oss<<"<O3vec";
      if(BASE::is_batched() && BASE::nbatch()>1) oss<<" b="<<BASE::nbatch();
      if(BASE::is_grid()) oss<<" grid="<<BASE::gdims();
      oss<<" tau="<<get_tau();
      if(get_dev()>0) oss<<" device="<<BASE::get_dev();
      oss<<">";
      return oss.str();
    }
    
    friend ostream& operator<<(ostream& stream, const O3vec& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 
