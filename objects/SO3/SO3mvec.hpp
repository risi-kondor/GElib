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


namespace GElib{

  typedef SO3vecB_array SO3mvec_base;

  class SO3mvec: public SO3mvec_base{
  public:

    using SO3mvec_base::SO3mvec_base;
    //using SO3vecB_array::SO3vecB_array;


    public:

    // ---- Constructors --------------------------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3mvec(const int b, const int k, const SO3type& tau, const FILLTYPE fill, const int _dev):
      SO3vecB_array(Gdims({b,k}),tau,fill,_dev){}
    //for(int l=0; l<tau.size(); l++)
    //parts.push_back(new SO3partB_array(Gdims({b,k}),l,tau[l],fill,_dev));
    //}

    
  public: // ---- Named constructors -------------------------------------------------------------------------

    
    static SO3mvec zero(const int b, const int k, const SO3type& tau, const int _dev=0){
      return SO3mvec(b,k,tau,cnine::fill_zero(),_dev);
    }
  
    static SO3mvec gaussian(const int b, const int k, const SO3type& tau, const int _dev=0){
      return SO3mvec(b,k,tau,cnine::fill_gaussian(),_dev);
    }

    static SO3mvec zeros_like(const SO3mvec& x){
      return SO3mvec::zero(x.getb(),x.getk(),x.get_tau(),x.get_dev());
    }

    static SO3mvec gaussian_like(const SO3mvec& x){
      return SO3mvec::gaussian(x.getb(),x.getk(),x.get_tau(),x.get_dev());
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
