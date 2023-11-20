
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibGpartE
#define _GElibGpartE

#include "GElib_base.hpp"
#include "diff_class.hpp"
#include "Ggroup.hpp"
#include "GirrepIx.hpp"
#include "GpartSpec.hpp"


namespace GElib{


  template<typename TYPE>
  class GpartE: public cnine::Ltensor<TYPE>,
		public cnine::diff_class<GpartE<TYPE> >{
  public:

    typedef cnine::Ltensor<TYPE> BASE;
    typedef cnine::diff_class<GpartE<TYPE> > DIFF_CLASS;

    typedef cnine::Gdims Gdims;

    using BASE::BASE;
    using BASE::arr;
    using BASE::dims;
    using BASE::strides;
    using BASE::dev;
    using BASE::labels;

    using BASE::dim;
    using BASE::device;

    using BASE::bgfused_view3;

    using BASE::is_batched;
    using BASE::nbatch;

    using BASE::is_grid;
    using BASE::gdims;
    using BASE::cell;

    using BASE::cdims;

#ifdef WITH_FAKE_GRAD
    using DIFF_CLASS::grad;
    using DIFF_CLASS::add_to_grad;
#endif 


    shared_ptr<Ggroup> G;
    shared_ptr<GirrepIx> ix;

    ~GpartE(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    //GpartE(): 
    //GpartE({1},cnine::DimLabels(),0,0){}

    //GpartE(const Ggroup& _G): G(_G){}


  public: // ---- GpartSpec -------------------------------------------------------------------------------


    template<typename SPEC>
    GpartE(const GpartSpecBase<SPEC>& g):
      BASE(g.get_dims(), g.get_labels(), g.get_fcode(), g.get_dev()),
      G(g.G), 
      ix(g.ix){}

    GpartSpec spec() const{
      return GpartSpec(dims,labels,dev,G,ix); // TODO 
    }


  public: // ---- Copying ------------------------------------------------------------------------------------

    
    GpartE(const GpartE& x):
      BASE(x), G(x.G), ix(x.ix){}

    GpartE(GpartE&& x):
      BASE(std::move(x)), G(x.G), ix(x.ix){}

    GpartE& operator=(const GpartE& x){
      (*this)=BASE::operator=(x);
      G=x.G;
      ix=x.ix;
      return *this;
    }

    GpartE copy() const{
      auto r=BASE::copy();
      r.G=G;
      r.ix=ix;
      return r;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int getl() const{
      return (dims(-2)-1)/2;
    }

    int getn() const{
      return dims(-1);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    //cnine::Ltensor<TYPE> operator*(const cnine::Ltensor<TYPE>& y) const=delete;

    GpartE<TYPE> operator*(const cnine::Ltensor<TYPE>& y) const{
      CNINE_ASSRT(y.ndims()==2);
      CNINE_ASSRT(y.dim(0)==dims(-1));
      GpartE<TYPE> R(spec().n(y.dim(1)));
      R.add_mprod(*this,y);
      return R;
  }


  public: // ---- CG-products --------------------------------------------------------------------------------


    void add_CGproduct(const GpartE& x, const GpartE& y, const int _offs=0){
      //G->add_CGproduct(*this,x,y,_offs);
      G->addCGproduct(bgfused_view3(),x.bgfused_view3(),y.bgfused_view3(),_offs);
    }

    void add_CGproduct_back0(const GpartE& g, const GpartE& y, const int _offs=0){
      //G->add_CGproduct_back0(*this,g,y,_offs);
      G->addCGproduct_back0(bgfused_view3(),g.bgfused_view3(),y.bgfused_view3(),_offs);
    }

    void add_CGproduct_back1(const GpartE& g, const GpartE& x, const int _offs=0){
      //G->add_CGproduct_back1(*this,g,g,_offs);
      G->addCGproduct_back1(bgfused_view3(),g.bgfused_view3(),x.bgfused_view3(),_offs);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::SO3partE";
    }

    string repr() const{
      ostringstream oss;
      oss<<"GpartE(G="<<G->repr()<<",";
      oss<<ix->repr()<<",";
      if(is_batched()) oss<<"b="<<nbatch()<<",";
      if(is_grid()) oss<<"grid="<<gdims()<<",";
      //oss<<"l="<<getl()<<",";
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

    friend ostream& operator<<(ostream& stream, const GpartE& x){
      stream<<x.str(); return stream;
    }

  };

  template<typename TYPE, typename IX>
  inline GpartE<TYPE> CGproduct(const GpartE<TYPE>& x, const GpartE<TYPE>& y, const IX& l){
    GpartE<TYPE> r(x.spec().irrep(x.G->new_irrep(l)).n(x.getn()*y.getn()));
    r.add_CGproduct(x,y);
    return r;
  }



}

#endif 


  /*
  inline GpartE<complex<float> > operator*
  (const GpartE<complex<float> >& x, const cnine::Ltensor<complex<float> >& y){
    CNINE_ASSRT(y.ndims()==2);
    CNINE_ASSRT(y.dim(0)==x.dims(-1));
    GpartE<complex<float> > R(x.spec().n(y.dim(1)));
    R.add_mprod(x,y);
    return R;
  }
  */
