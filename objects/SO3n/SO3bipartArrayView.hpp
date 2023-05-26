
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3bipartArrayView
#define _GElibSO3bipartArrayView

#include "GElib_base.hpp"
#include "BatchedTensorArrayView.hpp"
#include "TensorTemplates.hpp"
#include "SO3bipartView.hpp"
#include "SO3part_view.hpp"
#include "SO3partArrayView.hpp"

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
  class SO3bipartArrayView: public cnine::BatchedTensorArrayView<complex<RTYPE> >{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::Gindex Gindex;

    typedef cnine::BatchedTensorArrayView<complex<RTYPE> > BatchedTensorArrayView;
    typedef SO3bipartView<RTYPE> _SO3bipartView;
    
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


    SO3bipartArrayView(){}

    SO3bipartArrayView(const int b, const Gdims& _adims, const int l, const int n, const int _dev=0):
      BatchedTensorArrayView(b,_adims,Gdims({2*l+1,n}),_dev){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3bipartArrayView(const int b, const Gdims& _adims, const int l1, const int l2, const int n, const FILLTYPE& fill, const int _dev=0):
      BatchedTensorArrayView(b,_adims,Gdims({2*l1+1,2*l2+1,n}),fill,_dev){}


  public: // ---- Conversions --------------------------------------------------------------------------------


    //SO3bipartArrayView(const cnine::TensorArrayView<complex<RTYPE> >& x):
    //BatchedTensorArrayView(x){}

    //SO3bipartArrayView(const cnine::TensorView<complex<RTYPE> >& x):
    //BatchedTensorArrayView(x){
    //ak=ndims()-2;}

    SO3bipartArrayView(const BatchedTensorArrayView& x):
      BatchedTensorArrayView(x){}

    operator cnine::Ctensor4_view() const{
      return cnine::Ctensor4_view(arr.template ptr_as<RTYPE>(),{getb()*getN(),dims(-3),dims(-2),dims(-1)},
	{2*strides(-4),2*strides(-3),2*strides(-2),2*strides(-1)},1,device());
    }


  public: // ---- Access --------------------------------------------------------------------------------------

    
    int getl1() const{
      return (dims.back(2)-1)/2;
    }

    int getl2() const{
      return (dims.back(1)-1)/2;
    }

    int getn() const{
      return dims.back(0);
    }


    SO3bipartArrayView<RTYPE> batch(const int i) const{
      return SO3bipartArrayView<RTYPE>(arr+strides[0]*i,nadims(),dims.chunk(1).prepend(1),strides);
    }


    _SO3bipartView operator()(const int i0){
      CNINE_ASSRT(ak==2);
      return _SO3bipartView(arr+strides[1]*i0,get_ddims(),get_dstrides());
    }

    _SO3bipartView operator()(const int i0, const int i1){
      CNINE_ASSRT(ak==3);
      return _SO3bipartView(arr+strides[1]*i0+strides[2]*i1,get_ddims(),get_dstrides());
    }

    _SO3bipartView operator()(const int i0, const int i1, const int i2){
      CNINE_ASSRT(ak==4);
      return _SO3bipartView(arr+strides[1]*i0+strides[2]*i1+strides[3]*i2,get_ddims(),get_dstrides());
    }

    _SO3bipartView operator()(const Gindex& ix){
      CNINE_ASSRT(ix.size()==ak);
      return _SO3bipartView(arr+strides.chunk(1)(ix),get_ddims(),get_dstrides());
    }


    _SO3bipartView cell(const int i0){
      CNINE_ASSRT(ak==2);
      return _SO3bipartView(arr+strides[1]*i0,get_ddims(),get_dstrides());
    }

    _SO3bipartView cell(const int i0, const int i1){
      CNINE_ASSRT(ak==3);
      return _SO3bipartView(arr+strides[1]*i0+strides[2]*i1,get_ddims(),get_dstrides());
    }

    _SO3bipartView cell(const int i0, const int i1, const int i2){
      CNINE_ASSRT(ak==4);
      return _SO3bipartView(arr+strides[1]*i0+strides[2]*i1+strides[3]*i2,get_ddims(),get_dstrides());
    }

    _SO3bipartView cell(const Gindex& ix){
      CNINE_ASSRT(ix.size()==ak);
      return _SO3bipartView(arr+strides.chunk(1)(ix),get_ddims(),get_dstrides());
    }


  public: // ---- CG-products --------------------------------------------------------------------------------

    
  public: // ---- CG-transforms ------------------------------------------------------------------------------


    void add_CGtransform_to(const SO3partArrayView<RTYPE>& r, const int offs=0) const{
      SO3part_addCGtransformFn()(cnine::Ctensor3_view(r),cnine::Ctensor4_view(*this),offs);
    }

    void add_CGtransform_back(const SO3partArrayView<RTYPE>& r, const int offs=0) const{
      SO3part_addCGtransform_backFn()(cnine::Ctensor4_view(*this),cnine::Ctensor3_view(r),offs);
    }



  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "SO3bipartArrayView";
    }

    string repr(const string indent="") const{
      return "<GElib::SO3bipart(b="+to_string(getb())+",adims="+get_adims().str()+",l1="+to_string(getl1())+
	",l2="+to_string(getl2())+",n="+to_string(getn())+")>";
    }
    
    friend ostream& operator<<(ostream& stream, const SO3bipartArrayView& x){
      stream<<x.str(); return stream;
    }

  };

}


#endif 


 
