
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
//#include "TensorView.hpp"
#include "TensorVirtual.hpp"
#include "SO3partView.hpp"
#include "SO3templates.hpp"


namespace GElib{

  template<typename TYPE>
  class SO3part: public cnine::TensorVirtual<complex<TYPE>, SO3partView<TYPE> >,
		 public cnine::diff_class<SO3part<TYPE> >{
  public:

    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::Gdims Gdims;

    typedef cnine::TensorVirtual<complex<TYPE>, SO3partView<TYPE> > VTensor;
    typedef SO3partView<TYPE> SO3partView;
    typedef cnine::diff_class<SO3part<TYPE> > diff_class;

    using VTensor::VTensor;
    using VTensor::dims;
    using VTensor::operator*;

    using SO3partView::getl;
    using SO3partView::getn;
    using SO3partView::dim;

    using diff_class::grad;
    using diff_class::add_to_grad;


    ~SO3part(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    SO3part(const int b, const int l, const int n, const int _dev=0):
      SO3part(b,Gdims({2*l+1,n}),_dev){}


  public: // ---- Named constructors -------------------------------------------------------------------------

    
    static SO3part raw(const int l, const int n){
      return SO3part(1,Gdims({2*l+1,n}));}
    
    static SO3part zero(const int l, const int n){
      return SO3part(1,Gdims({2*l+1,n}),cnine::fill_zero());}
    
    static SO3part sequential(const int l, const int n){
      return SO3part(1,Gdims({2*l+1,n}),cnine::fill_sequential());}
    
    static SO3part gaussian(const int l, const int n){
      return SO3part(1,Gdims({2*l+1,n}),cnine::fill_gaussian());}

    
    static SO3part raw(const int b, const int l, const int n, const int _dev=0){
      return SO3part(b,Gdims({2*l+1,n}),_dev);}
    
    static SO3part zero(const int b, const int l, const int n, const int _dev=0){
      return SO3part(b,Gdims({2*l+1,n}),cnine::fill_zero(),_dev);}
    
    static SO3part sequential(const int b, const int l, const int n, const int _dev=0){
      return SO3part(b,Gdims({2*l+1,n}),cnine::fill_sequential(),_dev);}
    
    static SO3part gaussian(const int b, const int l, const int n, const int _dev=0){
      return SO3part(b,Gdims({2*l+1,n}),cnine::fill_gaussian(),_dev);}


    static SO3part* new_zeros_like(const SO3part& x){
      return new SO3part(x.getb(),Gdims({2*x.getl()+1,x.getn()}),cnine::fill_zero(),x.device());}


  public: // ---- Access -------------------------------------------------------------------------------------


    //int getl() const{
    //return (dims(0)-1)/2;
    //}

    //int getn() const{
    //return dims(1);
    //}

    bool is_F() const{
      return (dim(0)==dim(1));
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    
  public: // ---- CG-products --------------------------------------------------------------------------------

    


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::SO3part";
    }

    //string repr(const string indent="") const{
    //return "<GElib::SO3part(l="+to_string(getl())+",n="+to_string(getn())+")>";
    //}
    
    friend ostream& operator<<(ostream& stream, const SO3part& x){
      stream<<x.str(); return stream;
    }


  };


  template<typename TYPE>
  inline SO3part<TYPE> CGproduct(const SO3partView<TYPE>& x, const SO3partView<TYPE>& y, const int l){
    assert(l>=abs(x.getl()-y.getl()) && l<=x.getl()+y.getl());
    SO3part<TYPE> R=SO3part<TYPE>::zero(x.getb(),l,x.getn()*y.getn(),x.device());
    //add_CGproduct(R,x,y);
    R.add_CGproduct(x,y);
    return R;
    }

  template<typename TYPE>
  inline SO3part<TYPE> DiagCGproduct(const SO3partView<TYPE>& x, const SO3partView<TYPE>& y, const int l){
      assert(x.getn()==y.getn());
      assert(l>=abs(x.getl()-y.getl()) && l<=x.getl()+y.getl());
      SO3part<TYPE> R=SO3part<TYPE>::zero(x.getb(),l,x.getn(),x.device());
      add_DiagCGproduct(R,x,y);
      return R;
    }




}


#endif 
