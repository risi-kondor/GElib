
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3partArrayView
#define _GElibSO3partArrayView

#include "GElib_base.hpp"
#include "BatchedTensorArrayView.hpp"
#include "TensorTemplates.hpp"
#include "SO3partView.hpp"
#include "SO3part_view.hpp"

#include "SO3part_addCGproductFn.hpp"
#include "SO3part_addCGproduct_back0Fn.hpp"
#include "SO3part_addCGproduct_back1Fn.hpp"

#include "SO3part_addRCGproductFn.hpp"
#include "SO3part_addRCGproduct_back0Fn.hpp"
#include "SO3part_addRCGproduct_back1Fn.hpp"

#include "SO3part_addBlockedCGproductFn.hpp"
#include "SO3part_addBlockedCGproduct_back0Fn.hpp"
#include "SO3part_addBlockedCGproduct_back1Fn.hpp"


namespace GElib{

  template<typename RTYPE>
  class SO3partArrayView: public cnine::BatchedTensorArrayView<complex<RTYPE> >{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::Gindex Gindex;

    typedef cnine::BatchedTensorArrayView<complex<RTYPE> > BatchedTensorArrayView;
    typedef SO3partView<RTYPE> SO3partView;
    
    using BatchedTensorArrayView::arr;
    using BatchedTensorArrayView::dims;
    using BatchedTensorArrayView::strides;
    using BatchedTensorArrayView::dev;
    using BatchedTensorArrayView::ak;

    using BatchedTensorArrayView::BatchedTensorArrayView;
    //using BatchedTensorArrayView::operator=;
    using BatchedTensorArrayView::device;
    using BatchedTensorArrayView::ndims;
    using BatchedTensorArrayView::nadims;
    using BatchedTensorArrayView::nddims;
    using BatchedTensorArrayView::get_adims;
    using BatchedTensorArrayView::get_ddims;
    using BatchedTensorArrayView::getb;
    using BatchedTensorArrayView::get_astrides;
    using BatchedTensorArrayView::get_dstrides;
    using BatchedTensorArrayView::getN;
    using BatchedTensorArrayView::slice;

    
  public: // ---- Constructors for non-view child classes ---------------------------------------------------


    SO3partArrayView(const int b, const Gdims& _adims, const int l, const int n, const int _dev=0):
      BatchedTensorArrayView(b,_adims,Gdims({2*l+1,n}),_dev){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3partArrayView(const int b, const Gdims& _adims, const int l, const int n, const FILLTYPE& fill, const int _dev=0):
      BatchedTensorArrayView(b,_adims,Gdims({2*l+1,n}),fill,_dev){}



  public: // ---- Conversions --------------------------------------------------------------------------------


    //SO3partArrayView(const cnine::TensorArrayView<complex<RTYPE> >& x):
    //BatchedTensorArrayView(x){}

    //SO3partArrayView(const cnine::TensorView<complex<RTYPE> >& x):
    //BatchedTensorArrayView(x){
    //ak=ndims()-2;}

    SO3partArrayView(const BatchedTensorArrayView& x):
      BatchedTensorArrayView(x){}

    operator SO3part3_view() const{
      return SO3part3_view(arr.template ptr_as<RTYPE>(),{getb()*getN(),dims(-2),dims(-1)},
	{2*strides(-3),2*strides(-2),2*strides(-1)},1,device());
    }


  public: // ---- Access --------------------------------------------------------------------------------------

    
    int getl() const{
      return (dims.back(1)-1)/2;
    }

    int getn() const{
      return dims.back(0);
    }


    SO3partArrayView<RTYPE> batch(const int i) const{
      return SO3partArrayView<RTYPE>(arr+strides[0]*i,nadims(),dims.chunk(1).prepend(1),strides);
    }


    SO3partView operator()(const int i0){
      CNINE_ASSRT(ak==2);
      return SO3partView(arr+strides[1]*i0,get_ddims(),get_dstrides());
    }

    SO3partView operator()(const int i0, const int i1){
      CNINE_ASSRT(ak==3);
      return SO3partView(arr+strides[1]*i0+strides[2]*i1,get_ddims(),get_dstrides());
    }

    SO3partView operator()(const int i0, const int i1, const int i2){
      CNINE_ASSRT(ak==4);
      return SO3partView(arr+strides[1]*i0+strides[2]*i1+strides[3]*i2,get_ddims(),get_dstrides());
    }

    SO3partView operator()(const Gindex& ix){
      CNINE_ASSRT(ix.size()==ak);
      return SO3partView(arr+strides.chunk(1)(ix),get_ddims(),get_dstrides());
    }


    SO3partView cell(const int i0){
      CNINE_ASSRT(ak==2);
      return SO3partView(arr+strides[1]*i0,get_ddims(),get_dstrides());
    }

    SO3partView cell(const int i0, const int i1){
      CNINE_ASSRT(ak==3);
      return SO3partView(arr+strides[1]*i0+strides[2]*i1,get_ddims(),get_dstrides());
    }

    SO3partView cell(const int i0, const int i1, const int i2){
      CNINE_ASSRT(ak==4);
      return SO3partView(arr+strides[1]*i0+strides[2]*i1+strides[3]*i2,get_ddims(),get_dstrides());
    }

    SO3partView cell(const Gindex& ix){
      CNINE_ASSRT(ix.size()==ak);
      return SO3partView(arr+strides.chunk(1)(ix),get_ddims(),get_dstrides());
    }


  public: // ---- CG-products --------------------------------------------------------------------------------

    
    void add_CGproduct(const SO3partArrayView& x, const SO3partArrayView& y, const int _offs=0) const{
      cnine::reconcile_batched_array<SO3partArrayView>(*this,x,y,
	[&](const auto& r, const auto& x, const auto& y){SO3part_addCGproductFn()(r,x,y,_offs);},
	[&](const auto& r, const auto& x, const auto& y){SO3part_addRCGproductFn()(r,x,y,_offs);});
    }

    void add_CGproduct_back0(const SO3partArrayView& g, const SO3partArrayView& y, const int _offs=0){
      cnine::reconcile_batched_array<SO3partArrayView>(*this,g,y,
	[&](const auto& xg, const auto& g, const auto& y){SO3part_addCGproduct_back0Fn()(xg,g,y,_offs);},
	[&](const auto& xg, const auto& g, const auto& y){SO3part_addRCGproduct_back0Fn()(xg,g,y,_offs);});
    }

    void add_CGproduct_back1(const SO3partArrayView& g, const SO3partArrayView& x, const int _offs=0){
      cnine::reconcile_batched_array<SO3partArrayView>(*this,g,x,
	[&](const auto& yg, const auto& g, const auto& x){SO3part_addCGproduct_back1Fn()(yg,g,x,_offs);},
	[&](const auto& yg, const auto& g, const auto& x){SO3part_addRCGproduct_back1Fn()(yg,g,x,_offs);});
    }


    void add_DiagCGproduct(const SO3partArrayView& x, const SO3partArrayView& y, const int _offs=0) const{
      cnine::reconcile_batched_array<SO3partArrayView>(*this,x,y,
	[&](const auto& r, const auto& x, const auto& y){SO3part_addBlockedCGproductFn()(r,x,y,1,_offs);},
	[&](const auto& r, const auto& x, const auto& y){
	  GELIB_UNIMPL();
	  //SO3part_addRBlockedCGproductFn()(r,x,y,1,_offs);
	});
    }

    void add_DiagCGproduct_back0(const SO3partArrayView& g, const SO3partArrayView& y, const int _offs=0){
      cnine::reconcile_batched_array<SO3partArrayView>(*this,g,y,
	[&](const auto& xg, const auto& g, const auto& y){SO3part_addBlockedCGproduct_back0Fn()(xg,g,y,1,_offs);},
	[&](const auto& xg, const auto& g, const auto& y){
	  GELIB_UNIMPL();
	  //SO3part_addRBlockedCGproduct_back0Fn()(xg,g,y,1,_offs);
	});
    }

    void add_DiagCGproduct_back1(const SO3partArrayView& g, const SO3partArrayView& x, const int _offs=0){
      cnine::reconcile_batched_array<SO3partArrayView>(*this,g,x,
	[&](const auto& yg, const auto& g, const auto& x){SO3part_addBlockedCGproduct_back1Fn()(yg,g,x,1,_offs);},
	[&](const auto& yg, const auto& g, const auto& x){
	  GELIB_UNIMPL();
	  //SO3part_addRBlockedCGproduct_back1Fn()(yg,g,x,1,_offs);
	});
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "SO3partArrayView";
    }

    string repr(const string indent="") const{
      return "<GElib::SO3part(b="+to_string(getb())+",adims="+get_adims().str()+",l="+to_string(getl())+",n="+to_string(getn())+")>";
    }
    
    friend ostream& operator<<(ostream& stream, const SO3partArrayView& x){
      stream<<x.str(); return stream;
    }

  };

}


#endif 


 
