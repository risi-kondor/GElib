// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2024, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3part
#define _SO3part

#include "Gpart.hpp"


namespace GElib{


  template<typename TYPE>
  class SO3part: public Gpart<complex<TYPE> >{
  public:

    typedef Gpart<complex<TYPE> > BASE;
    typedef cnine::Ltensor<complex<TYPE> > TENSOR;
    typedef int IrrepIx; 

    typedef cnine::Gdims Gdims;

    using TENSOR::get_dev;

    using BASE::unroller;
    using BASE::get_nc;


  public: // ---- Constructors -------------------------------------------------------------------------------


    SO3part(){}

    SO3part(const int _b, const int _l, const int _nc, const int _fcode=0, const int _dev=0):
      BASE(_b,2*_l+1,_nc,_fcode,_dev){}

    SO3part(const int _b, const Gdims& _gdims, const int _l, const int _nc, const int _fcode=0, const int _dev=0):
      BASE(_b,_gdims,2*_l+1,_nc,_fcode,_dev){}


  public: // ---- Named parameter constructors ---------------------------------------------------------------


    //template<typename... Args>
    //SO3part(const IrrepArgument& x, const Args&... args){
    //unroll(x,args...);
    //}

    template<typename... Args>
    SO3part(const Args&... args){
      typename BASE::vparams v;
      unroller(v,args...);
      int ell=any_cast<int>(v.ell);
      BASE::reset(v.b,v.gdims,2*ell+1,v.nc,v.fcode,v.dev);
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    SO3part(const SO3part& x):
      BASE(x){}

    //SO3part(const SO3part&& x):
    //BASE(x){}


  public: // ---- Access -------------------------------------------------------------------------------------

    
    int getl() const{
      return (TENSOR::dims(-2)-1)/2;
    }


  public: // ---- I/O -------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::SO3part";
    }

    string repr() const{
      ostringstream oss;
      oss<<"<SO3part";
      if(BASE::is_batched() && BASE::nbatch()>1) oss<<" b="<<BASE::nbatch();
      if(BASE::is_grid()) oss<<" grid="<<BASE::gdims();
      oss<<" l="<<getl();
      oss<<" nc="<<get_nc();
      if(get_dev()>0) oss<<" device="<<get_dev();
      oss<<">";
      return oss.str();
    }
    
    string to_print(const string indent="") const{
      ostringstream oss;
      oss<<indent<<repr()<<":"<<endl;
      oss<<TENSOR::str(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const SO3part& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 

