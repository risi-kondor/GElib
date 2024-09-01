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
#include "SO3group.hpp"
#include "SO3type.hpp"
#include "SO3part_addCGproductFn.hpp"

namespace GElib{


  template<typename TYPE>
  class SO3part: public Gpart<SO3part<TYPE>,complex<TYPE> >{
  public:

    typedef Gpart<SO3part<TYPE>,complex<TYPE> > BASE;
    typedef cnine::TensorView<complex<TYPE> > TENSOR;

    typedef SO3group GROUP;
    typedef int IRREP_IX; 
    typedef SO3type GTYPE;

    typedef cnine::Gdims Gdims;

    using TENSOR::get_dev;
    using TENSOR::ndims;
    using TENSOR::dim;
    using TENSOR::dims;

    using BASE::unroller;
    using BASE::getn;
    using BASE::dominant_batch;
    using BASE::dominant_gdims;
    using BASE::co_promote;


  public: // ---- Constructors -------------------------------------------------------------------------------


    SO3part(){}

    SO3part(const int _b, const int _l, const int _nc, const int _fcode=0, const int _dev=0):
      BASE(_b,2*_l+1,_nc,_fcode,_dev){}

    SO3part(const int _b, const Gdims& _gdims, const int _l, const int _nc, const int _fcode=0, const int _dev=0):
      BASE(_b,_gdims,2*_l+1,_nc,_fcode,_dev){}


  public: // ---- Named parameter constructors ---------------------------------------------------------------


    template<typename ARG0, typename... Args, 
	     typename = typename std::enable_if<
    std::is_same<IrrepArgument, ARG0>::value || 
    std::is_same<cnine::BatchArgument, ARG0>::value ||
    std::is_same<cnine::GridArgument, ARG0>::value ||
    std::is_same<cnine::ChannelsArgument, ARG0>::value ||
    std::is_same<cnine::FillArgument, ARG0>::value ||
    std::is_same<cnine::DeviceArgument, ARG0>::value, ARG0>::type>
    SO3part(const ARG0& arg0, const Args&... args){
      typename BASE::vparams v;
      unroller(v,arg0,args...);
      if(v.ell.has_value()==false) 
	throw std::invalid_argument("GElib error: constructor of SO3part must have an irrep argument.");
      int ell=any_cast<int>(v.ell);
      BASE::reset(v.b,v.gdims,2*ell+1,v.nc,v.fcode,v.dev);
    }


  public: // ---- Factory methods -------------------------------------------------------------------------------------


    SO3part zeros_like() const{
      return BASE::zeros_like();
    }

    SO3part zeros_like(const int l) const{
      return BASE::zeros_like(2*l+1);
    }

    SO3part zeros_like(const int l, const int n) const{
      return BASE::zeros_like(2*l+1,n);
    }


  public: // ---- Copying ------------------------------------------------------------------------------------



  public: // ---- Conversions ---------------------------------------------------------------------------------


    SO3part(const TENSOR& x):
      BASE(x){
      GELIB_ASSRT(ndims()>=3);
      GELIB_ASSRT(dims(1)%2==1);
    }


  public: // ---- Transport -----------------------------------------------------------------------------------


    SO3part(const SO3part& x, const int _dev){
      return SO3partB(TENSOR(x,_dev));
    }


  public: // ---- Access -------------------------------------------------------------------------------------

    
    int getl() const{
      return (TENSOR::dims(-2)-1)/2;
    }


  public: // ---- CG-products --------------------------------------------------------------------------------

    
    void add_CGproduct(const SO3part& x, const SO3part& y, const int _offs=0){
      auto [x0,y0]=x.co_promote(y);
      SO3part_addCGproductFn<SO3part,TYPE>()(*this,x,y,_offs);
    }

    void add_CGproduct_back0(const SO3part& g, const SO3part& y, const int _offs=0){
      //SO3part_addCGproduct_back0Fn()(*this,g,y,_offs);
    }

    void add_CGproduct_back1(const SO3part& g, const SO3part& x, const int _offs=0){
      //SO3part_addCGproduct_back1Fn()(*this,g,x,_offs);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::SO3part";
    }

    string repr() const{
      ostringstream oss;
      oss<<"<SO3part";
      if(BASE::is_batched()) oss<<" b="<<BASE::getb();
      if(BASE::is_grid()) oss<<" grid="<<BASE::gdims();
      oss<<" l="<<getl();
      oss<<" nc="<<getn();
      if(get_dev()>0) oss<<" device="<<get_dev();
      oss<<">";
      return oss.str();
    }
    
    friend ostream& operator<<(ostream& stream, const SO3part& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 


    //string to_print(const string indent="") const{
    //ostringstream oss;
    //oss<<indent<<repr()<<":"<<endl;
    //oss<<BASE::str(indent+"  ");
    //return oss.str();
    //}

