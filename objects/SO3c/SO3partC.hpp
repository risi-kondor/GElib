
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3partC
#define _GElibSO3partC

#include "GElib_base.hpp"
#include "Ltensor.hpp"
#include "SO3partSpec.hpp"
#include "diff_class.hpp"
#include "WorkStreamLoop.hpp"

#include "SO3part_addCGproductFn.hpp"


namespace GElib{


  template<typename TYPE>
  class SO3part: public cnine::Ltensor<complex<TYPE> >,
		 public cnine::diff_class<SO3part<TYPE> >{
  public:

    typedef cnine::Ltensor<complex<TYPE> > BASE;
    typedef cnine::diff_class<SO3part<TYPE> > diff_class;

    typedef cnine::Gdims Gdims;

    using BASE::BASE;
    using BASE::arr;
    using BASE::dims;
    using BASE::strides;
    using BASE::dev;

    using BASE::dim;
    using BASE::device;
    using BASE::is_batched;
    using BASE::nbatch;


#ifdef WITH_FAKE_GRAD
    using diff_class::grad;
    using diff_class::add_to_grad;
#endif 

    ~SO3part(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    SO3part(): 
      SO3part({1},cnine::DimLabels(),0,0){}

    SO3part(const SO3partSpec<TYPE>& g):
      SO3part(g.get_dims(), g.get_labels(), g.get_fcode(), g.get_dev()){}

    static SO3partSpec<TYPE> make() {return SO3partSpec<TYPE>();}
    static SO3partSpec<TYPE> raw() {return SO3partSpec<TYPE>().raw();}
    static SO3partSpec<TYPE> zero() {return SO3partSpec<TYPE>().zero();}
    static SO3partSpec<TYPE> sequential() {return SO3partSpec<TYPE>().sequential();}
    static SO3partSpec<TYPE> gaussian() {return SO3partSpec<TYPE>().gaussian();}

    SO3partSpec<TYPE> spec() const{
      return BASE::spec();
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


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

    


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::SO3part";
    }

    string repr() const{
      ostringstream oss;
      oss<<"SO3part[";
      if(is_batched()) oss<<"nbatch="<<nbatch()<<",";
      //if(_narray>0) oss<<"blocks="<<adims(dims)<<",";
      oss<<"l="<<getl()<<",";
      oss<<"n="<<getn()<<",";
      oss<<"\b]"<<"["<<dev<<"]";
      return oss.str();
    }
    
    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<repr()<<":"<<endl;
      oss<<BASE::to_string(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const SO3part& x){
      stream<<x.str(); return stream;
    }


  };


  template<typename TYPE>
  inline SO3part<TYPE> operator*(const SO3part<TYPE>& x, const cnine::Ltensor<complex<TYPE> >& y){
    CNINE_ASSRT(y.ndims()==2);
    CNINE_ASSRT(y.dim(0)==x.dims(-1));
    SO3part<TYPE> R(x.spec().channels(y.dim(1)));
    R.add_mprod(x,y);
    return R;
  }


  template<typename TYPE>
  inline SO3part<TYPE> CGproduct(const SO3part<TYPE>& x, const SO3part<TYPE>& y, const int l){
    assert(l>=abs(x.getl()-y.getl()) && l<=x.getl()+y.getl());
    SO3part<TYPE> r=SO3part<TYPE>::zero().l(l).n(x.getn()*y.getn()).dev(x.dev);
    SO3part_addCGproductFn()(r.view3(),x.view3(),y.view3());
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
