
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3bipart
#define _GElibSO3bipart

#include "GElib_base.hpp"
//#include "TensorView.hpp"
#include "TensorVirtual.hpp"
#include "diff_class.hpp"
#include "SO3bipartView.hpp"
#include "SO3partC.hpp"
#include "SO3templates.hpp"


namespace GElib{

  template<typename TYPE>
  class SO3bipart: public cnine::TensorVirtual<complex<TYPE>, SO3bipartView<TYPE> >,
		 public cnine::diff_class<SO3bipart<TYPE> >{
  public:

    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::Gdims Gdims;

    typedef cnine::TensorVirtual<complex<TYPE>, SO3bipartView<TYPE> > VTensor;
    // typedef SO3bipartView<TYPE> Pview;
    typedef cnine::diff_class<SO3bipart<TYPE> > diff_class;

    using VTensor::VTensor;
    using VTensor::dims;
    using VTensor::operator*;

    using SO3bipartView<TYPE>::getl1;
    using SO3bipartView<TYPE>::getl2;
    using SO3bipartView<TYPE>::getn;
    using SO3bipartView<TYPE>::dim;
    using SO3bipartView<TYPE>::str;

#ifdef WITH_FAKE_GRAD
    using diff_class::grad;
    using diff_class::add_to_grad;
#endif 

    ~SO3bipart(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    SO3bipart(const int b, const int l1, const int l2, const int n, const int _dev=0):
      SO3bipart(b,Gdims({2*l1+1,2*l2+1,n}),_dev){}


  public: // ---- Named constructors -------------------------------------------------------------------------

    
    static SO3bipart raw(const int l1, const int l2, const int n){
      return SO3bipart(1,Gdims({2*l1+1,2*l2+1,n}));}
    
    static SO3bipart zero(const int l1, const int l2, const int n){
      return SO3bipart(1,Gdims({2*l1+1,2*l2+1,n}),cnine::fill_zero());}
    
    static SO3bipart sequential(const int l1, const int l2, const int n){
      return SO3bipart(1,Gdims({2*l1+1,2*l2+1,n}),cnine::fill_sequential());}
    
    static SO3bipart gaussian(const int l1, const int l2, const int n){
      return SO3bipart(1,Gdims({2*l1+1,2*l2+1,n}),cnine::fill_gaussian());}

    
    static SO3bipart raw(const int b, const int l1, const int l2, const int n, const int _dev=0){
      return SO3bipart(b,Gdims({2*l1+1,2*l2+1,n}),_dev);}
    
    static SO3bipart zero(const int b, const int l1, const int l2, const int n, const int _dev=0){
      return SO3bipart(b,Gdims({2*l1+1,2*l2+1,n}),cnine::fill_zero(),_dev);}
    
    static SO3bipart sequential(const int b, const int l1, const int l2, const int n, const int _dev=0){
      return SO3bipart(b,Gdims({2*l1+1,2*l2+1,n}),cnine::fill_sequential(),_dev);}
    
    static SO3bipart gaussian(const int b, const int l1, const int l2, const int n, const int _dev=0){
      return SO3bipart(b,Gdims({2*l1+1,2*l2+1,n}),cnine::fill_gaussian(),_dev);}


    static SO3bipart* new_zeros_like(const SO3bipart& x){
      return new SO3bipart(x.getb(),Gdims({2*x.getl1()+1,2*x.getl2()+1,x.getn()}),cnine::fill_zero(),x.device());}


  public: // ---- Access -------------------------------------------------------------------------------------


    //int getl() const{
    //return (dims(0)-1)/2;
    //}

    //int getn() const{
    //return dims(1);
    //}

    //bool is_F() const{
    //return (dim(0)==dim(1));
    //}


  public: // ---- Operations ---------------------------------------------------------------------------------


    
  public: // ---- CG-transforms ------------------------------------------------------------------------------
    


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::SO3bipart";
    }

    //string repr(const string indent="") const{
    //return "<GElib::SO3bipart(l="+to_string(getl())+",n="+to_string(getn())+")>";
    //}
    
    friend ostream& operator<<(ostream& stream, const SO3bipart& x){
      stream<<x.str(); return stream;
    }


  };


  template<typename TYPE>
  inline SO3part<TYPE> CGtransform(const SO3bipartView<TYPE>& x, const int l){
    assert(l>=abs(x.getl1()-x.getl2()) && l<=x.getl1()+x.getl2());
    SO3part<TYPE> r=SO3part<TYPE>::zero(x.getb(),l,x.getn(),x.device());
    x.add_CGtransform_to(r);
    return r;
  }





}


#endif 
