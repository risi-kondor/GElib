
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3partD
#define _GElibSO3partD

#include "GElib_base.hpp"
#include "Ltensor.hpp"
#include "SO3group.hpp"
#include "SO3partSpec.hpp"
#include "diff_class.hpp"
#include "WorkStreamLoop.hpp"

#include "SO3part_addCGproductFn.hpp"
#include "SO3part_addCGproduct_back0Fn.hpp"
#include "SO3part_addCGproduct_back1Fn.hpp"

#include "SO3part_addBlockedCGproductFn.hpp"
#include "SO3part_addBlockedCGproduct_back0Fn.hpp"
#include "SO3part_addBlockedCGproduct_back1Fn.hpp"


namespace GElib{


  template<typename TYPE>
  class SO3partD: public cnine::Ltensor<complex<TYPE> >,
		 public cnine::diff_class<SO3partD<TYPE> >{
  public:

    typedef cnine::Ltensor<complex<TYPE> > BASE;
    typedef cnine::diff_class<SO3partD<TYPE> > DIFF_CLASS;

    typedef cnine::Gdims Gdims;

    using BASE::BASE;
    using BASE::arr;
    using BASE::dims;
    using BASE::strides;
    using BASE::dev;

    using BASE::dim;
    using BASE::device;

    using BASE::bgfused_view3;

    using BASE::is_batched;
    using BASE::nbatch;

    using BASE::is_grid;
    using BASE::gdims;
    using BASE::cell;

#ifdef WITH_FAKE_GRAD
    using DIFF_CLASS::grad;
    using DIFF_CLASS::add_to_grad;
#endif 

    ~SO3partD(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    SO3partD(): 
      SO3partD({1},cnine::DimLabels(),0,0){}


  public: // ---- SO3partSpec -------------------------------------------------------------------------------


    SO3partD(const SO3partSpec<TYPE>& g):
      SO3partD(g.get_dims(), g.get_labels(), g.get_fcode(), g.get_dev()){}

    static SO3partSpec<TYPE> make() {return SO3partSpec<TYPE>();}
    static SO3partSpec<TYPE> raw() {return SO3partSpec<TYPE>().raw();}
    static SO3partSpec<TYPE> zero() {return SO3partSpec<TYPE>().zero();}
    static SO3partSpec<TYPE> sequential() {return SO3partSpec<TYPE>().sequential();}
    static SO3partSpec<TYPE> gaussian() {return SO3partSpec<TYPE>().gaussian();}

    SO3partSpec<TYPE> spec() const{
      return BASE::spec();
    }


  public: // ---- Copying ------------------------------------------------------------------------------------

    
    SO3partD(const SO3partD& x):
      BASE(x){}

    SO3partD(SO3partD&& x):
      BASE(std::move(x)){}

    SO3partD& operator=(const SO3partD& x){
      (*this)=BASE::operator=(x);
      return *this;
    }

    SO3partD copy() const{
      return BASE::copy();
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    SO3partD(const BASE& x):
      BASE(x){}

    cnine::Ctensor3_view view3() const{
      if(is_batched()) return cnine::TensorView<complex<TYPE> >::view3();
      else return unsqueeze0(cnine::TensorView<complex<TYPE> >::view2());
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int getl() const{
      return (dims(-2)-1)/2;
    }

    int getn() const{
      return dims(-1);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    
  public: // ---- CG-products --------------------------------------------------------------------------------


    void add_CGproduct(const SO3partD& x, const SO3partD& y, const int _offs=0){
      SO3part_addCGproductFn()(bgfused_view3(),x.bgfused_view3(),y.bgfused_view3(),_offs);
    }

    void add_CGproduct_back0(const SO3partD& g, const SO3partD& y, const int _offs=0){
      SO3part_addCGproduct_back0Fn()(bgfused_view3(),g.bgfused_view3(),y.bgfused_view3(),_offs);
    }

    void add_CGproduct_back1(const SO3partD& g, const SO3partD& x, const int _offs=0){
      SO3part_addCGproduct_back1Fn()(bgfused_view3(),g.bgfused_view3(),x.bgfused_view3(),_offs);
    }


    void add_DiagCGproduct(const SO3partD& x, const SO3partD& y, const int _offs=0){
      SO3part_addBlockedCGproductFn()(bgfused_view3(),x.bgfused_view3(),y.bgfused_view3(),1,_offs);
    }

    void add_DiagCGproduct_back0(const SO3partD& g, const SO3partD& y, const int _offs=0){
      SO3part_addBlockedCGproduct_back0Fn()(bgfused_view3(),g.bgfused_view3(),y.bgfused_view3(),1,_offs);
    }

    void add_DiagCGproduct_back1(const SO3partD& g, const SO3partD& x, const int _offs=0){
      SO3part_addBlockedCGproduct_back1Fn()(bgfused_view3(),g.bgfused_view3(),x.bgfused_view3(),1,_offs);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::SO3partD";
    }

    string repr() const{
      ostringstream oss;
      oss<<"SO3partD(";
      if(is_batched()) oss<<"b="<<nbatch()<<",";
      if(is_grid()) oss<<"grid="<<gdims()<<",";
      oss<<"l="<<getl()<<",";
      oss<<"n="<<getn()<<",";
      if(dev>0) oss<<"dev="<<dev<<",";
      oss<<"\b)";
      return oss.str();
    }
    
    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<repr()<<":"<<endl;
      oss<<BASE::to_string(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const SO3partD& x){
      stream<<x.str(); return stream;
    }


  };


  template<typename TYPE>
  inline SO3partD<TYPE> operator*(const SO3partD<TYPE>& x, const cnine::Ltensor<complex<TYPE> >& y){
    CNINE_ASSRT(y.ndims()==2);
    CNINE_ASSRT(y.dim(0)==x.dims(-1));
    SO3partD<TYPE> R(x.spec().channels(y.dim(1)));
    R.add_mprod(x,y);
    return R;
  }


  template<typename TYPE>
  inline SO3partD<TYPE> CGproduct(const SO3partD<TYPE>& x, const SO3partD<TYPE>& y, const int l){
    assert(l>=abs(x.getl()-y.getl()) && l<=x.getl()+y.getl());
    SO3partD<TYPE> r(x.spec().l(l).n(x.getn()*y.getn()));
    r.add_CGproduct(x,y);
    return r;
  }

  /*
  template<typename TYPE>
  inline SO3part<TYPE> DiagCGproduct(const BASE& x, const BASE& y, const int l){
      assert(x.getn()==y.getn());
      assert(l>=abs(x.getl()-y.getl()) && l<=x.getl()+y.getl());
      SO3part<TYPE> R=SO3part<TYPE>::zero(x.getb(),l,x.getn(),x.device());
      add_DiagCGproduct(R,x,y);
      return R;
    }

  template<typename TYPE>
  inline SO3part<TYPE> StreamingCGproduct(const BASE& x, const BASE& y, const int l, const int dev=1){
    assert(l>=abs(x.getl()-y.getl()) && l<=x.getl()+y.getl());
    cnine::StreamingBlock bl(dev);
    SO3part<TYPE> R=SO3part<TYPE>::zero(x.getb(),l,x.getn()*y.getn(),x.device());
    R.add_CGproduct(x,y);
    return R;
    }
  */



}


#endif 
