// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3mvec
#define _SO3mvec

#include "GElib_base.hpp"
#include "SO3vecB_array.hpp"
#include "CtensorEinsumFn.hpp"
#include "FakeGrad.hpp"

namespace GElib{

  typedef SO3vecB_array SO3mvec_base;

  class SO3mvec: public SO3mvec_base 
#ifdef WITH_FAKE_GRAD
	       ,public cnine::FakeGrad<SO3mvec>
#endif
{
  public:

  #ifdef WITH_FAKE_GRAD
  ~SO3mvec(){
    if(!is_view) delete grad;
  }
  #endif


    using SO3mvec_base::SO3mvec_base;
    //using SO3vecB_array::SO3vecB_array;


    public: // ---- Constructors ------------------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3mvec(const int b, const int k, const SO3type& tau, const FILLTYPE fill, const int _dev):
      SO3vecB_array(Gdims({b,k}),tau,fill,_dev){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3mvec(const int b, const int k, const int maxl, const FILLTYPE fill, const int _dev):
      SO3vecB_array(Gdims({b,k}),maxl,fill,_dev){}

    
  public: // ---- Named constructors -------------------------------------------------------------------------

    
    static SO3mvec raw(const int k, const SO3type& tau, const int _dev=0){
      return SO3mvec(1,k,tau,cnine::fill_raw(),_dev);}
    static SO3mvec raw(const int b, const int k, const SO3type& tau, const int _dev=0){
      return SO3mvec(b,k,tau,cnine::fill_raw(),_dev);}
  
    static SO3mvec zero(const int k, const SO3type& tau, const int _dev=0){
      return SO3mvec(1,k,tau,cnine::fill_zero(),_dev);}
    static SO3mvec zero(const int b, const int k, const SO3type& tau, const int _dev=0){
      return SO3mvec(b,k,tau,cnine::fill_zero(),_dev);}
  
    static SO3mvec gaussian(const int k, const SO3type& tau, const int _dev=0){
      return SO3mvec(1,k,tau,cnine::fill_gaussian(),_dev);}
    static SO3mvec gaussian(const int b, const int k, const SO3type& tau, const int _dev=0){
      return SO3mvec(b,k,tau,cnine::fill_gaussian(),_dev);}


    // ---- Fourier constructors -----------------------------------------------------------------------------


    static SO3mvec Fraw(const int k, const int maxl, const int _dev=0){
      return SO3mvec(1,k,maxl,cnine::fill_raw(),_dev);}
    static SO3mvec Fraw(const int b, const int k, const int maxl, const int _dev=0){
      return SO3mvec(b,k,maxl,cnine::fill_raw(),_dev);}

    static SO3mvec Fzero(const int k, const int maxl, const int _dev=0){
      return SO3mvec(1,k,maxl,cnine::fill_zero(),_dev);}
    static SO3mvec Fzero(const int b, const int k, const int maxl, const int _dev=0){
      return SO3mvec(b,k,maxl,cnine::fill_zero(),_dev);}

    static SO3mvec Fgaussian(const int k, const int maxl, const int _dev=0){
      return SO3mvec(1,k,maxl,cnine::fill_gaussian(),_dev);}
    static SO3mvec Fgaussian(const int b, const int k, const int maxl, const int _dev=0){
      return SO3mvec(b,k,maxl,cnine::fill_gaussian(),_dev);}

  
    // ---- Like constructors --------------------------------------------------------------------------------


    static SO3mvec zeros_like(const SO3mvec& x){
      return SO3mvec::zero(x.getb(),x.getk(),x.get_tau(),x.get_dev());}

    static SO3mvec gaussian_like(const SO3mvec& x){
      return SO3mvec::gaussian(x.getb(),x.getk(),x.get_tau(),x.get_dev());}



public: // ---- Copying --------------------------------------------------------------------------------------


  SO3mvec(const SO3mvec& x):
    SO3mvec_base(x){}

  SO3mvec(SO3mvec&& x):
    SO3mvec_base(std::move(x)){}

  SO3mvec& operator=(const SO3mvec& x){
    SO3mvec_base::operator=(x);
    return *this;
  }

  SO3mvec& operator=(SO3mvec&& x){
    SO3mvec_base::operator=(std::move(x));
    return *this;
  }


public: // ---- Views ----------------------------------------------------------------------------------------
  

  SO3mvec view(){
    SO3mvec R=SO3mvec_base::view();
    #ifdef WITH_FAKE_GRAD
    if(grad) R.grad=new SO3mvec(grad->view());
    #endif 
    return R;
   }


   public: // ---- Conversions --------------------------------------------------------------------------------


    SO3mvec(const SO3mvec_base& x):
      SO3mvec_base(x){}

    SO3mvec(SO3mvec_base&& x):
      SO3mvec_base(std::move(x)){}


  public: // ---- Access -------------------------------------------------------------------------------------


    int getb() const{
      return get_adims()(0);
    }

    int getk() const{
      return get_adims()(1);
    }


  public: // ---- Experimental -------------------------------------------------------------------------------


    #ifdef WITH_FAKE_GRAD
    void add_to_part_of_grad(const int l, const SO3partB_array& x){
      if(!grad) grad=new SO3mvec(SO3mvec::zeros_like(*this));
      grad->parts[l]->add(x);
    }
    #endif


  public: // ---- Products -----------------------------------------------------------------------------------


    SO3mvec operator*(const cnine::CtensorPackObj& y) const{
      assert(y.tensors.size()==parts.size());
      SO3type tau;
      for(int l=0; l<parts.size(); l++){
	auto& w=*y.tensors[l];
	assert(w.get_ndims()==3);
	assert(w.get_dim(0)==getk());
	assert(w.get_dim(1)==parts[l]->getn());
	tau.push_back(w.get_dim(2));
      }
      SO3mvec R=SO3mvec::zero(getb(),getk(),tau,get_dev());
      R.add_mprod(*this,y);
      return R;
    }


    void add_mprod(const SO3mvec& x, const cnine::CtensorPackObj& y){
      CNINE_DEVICE_SAMEB(x);
      CNINE_DEVICE_SAMEB(y);
      assert(x.getk()==getk());
      assert(x.parts.size()==y.tensors.size());
      assert(x.parts.size()<=parts.size());
      cnine::CtensorEinsumFn<float> fn("adbi,dic->adbc");
      for(int l=0; l<x.parts.size(); l++)
	fn(parts[l]->viewx(),x.parts[l]->viewx(),y.tensors[l]->viewx());
    }


    void add_mprod_back0(const SO3mvec& g, const cnine::CtensorPackObj& y){
      CNINE_DEVICE_SAMEB(g);
      CNINE_DEVICE_SAMEB(y);
      assert(parts.size()==y.tensors.size());
      assert(parts.size()==g.parts.size());
      cnine::CtensorEinsumFn<float> fn("adbc,dic*->adbi");
      for(int l=0; l<parts.size(); l++)
	fn(parts[l]->viewx(),g.parts[l]->viewx(),y.tensors[l]->viewx());
    }


    void add_mprod_back1_into(cnine::CtensorPackObj& yg, const SO3mvec& x) const{
      CNINE_DEVICE_SAMEB(yg);
      CNINE_DEVICE_SAMEB(x);
      assert(parts.size()==yg.tensors.size());
      assert(parts.size()==x.parts.size());
      cnine::CtensorEinsumFn<float> fn("adbi*,adbc->dic");
      for(int l=0; l<parts.size(); l++)
	fn(yg.tensors[l]->viewx(),x.parts[l]->viewx(),parts[l]->viewx());
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------



  };


  // ---- Post-class functions -------------------------------------------------------------------------------


  inline SO3mvec CGproduct(const SO3mvec& x, const SO3mvec& y, const int maxl=-1){
    return x.CGproduct(y,maxl);
  }

  inline SO3mvec CGsquare(const SO3mvec& x, const int maxl=-1){
    return x.CGsquare(maxl);
  }

  inline SO3mvec Fproduct(const SO3mvec& x, const SO3mvec& y, const int maxl=-1){
    return x.Fproduct(y,maxl);
  }

  inline SO3mvec Fmodsq(const SO3mvec& x, const int maxl=-1){
    return x.Fmodsq(maxl);
  }

}

#endif 
