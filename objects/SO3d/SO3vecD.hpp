// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3vecD
#define _GElibSO3vecD

#include "GElib_base.hpp"
#include "GvecD.hpp"
#include "diff_class.hpp"
#include "SO3group.hpp"
#include "SO3partD.hpp"
#include "SO3vecSpec.hpp"

namespace GElib{

  template<typename TYPE>
  class SO3vecD: public GvecD<SO3group,TYPE>, //SO3partD<TYPE>>, 
		public cnine::diff_class<SO3vecD<TYPE> >{
  public:

    typedef GvecD<SO3group,TYPE> BASE;
    typedef cnine::diff_class<SO3vecD<TYPE> > DIFF_CLASS;

    typedef cnine::Gdims Gdims;

    using BASE::BASE;
    using BASE::parts;


    #ifdef WITH_FAKE_GRAD
    using DIFF_CLASS::grad;
    using DIFF_CLASS::add_to_grad;
    #endif 

    ~SO3vecD(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    //SO3vecD(const int _b, const cnine::Gdims& _gdims, ){}


  public: // ---- SO3vecSpec -------------------------------------------------------------------------------


    SO3vecD(const SO3vecSpec<TYPE>& spec):
      BASE(spec){
      for(int l=0; l<spec._tau.size(); l++)
	parts[l]=new SO3partD(SO3partSpec<TYPE>(spec).l(l).n(spec._tau(l)));
    }

    static SO3vecSpec<TYPE> raw() {return SO3vecSpec<TYPE>().raw();}
    static SO3vecSpec<TYPE> zero() {return SO3vecSpec<TYPE>().zero();}
    static SO3vecSpec<TYPE> sequential() {return SO3vecSpec<TYPE>().sequential();}
    static SO3vecSpec<TYPE> gaussian() {return SO3vecSpec<TYPE>().gaussian();}

    SO3vecSpec<TYPE> spec() const{
      return BASE::spec();
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    SO3vecD(const SO3vecD& x):
      BASE(x){}

    SO3vecD(SO3vecD&& x):
      BASE(std::move(x)){}

    SO3vecD& operator=(const SO3vecD& x){
      (*this)=BASE::operator=(x);
      return *this;
    }

    SO3vecD copy() const{
      return BASE::copy();
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    static string classname(){
      return "GElib::SO3vecD";
    }


  };


  // ---- Functions ------------------------------------------------------------------------------------------


  /*
  template<typename RTYPE>
  inline SO3vec<RTYPE> operator+(const SO3vec<RTYPE>& x, const SO3vec<RTYPE>& y){
    SO3vec<RTYPE> r(x);
    r.add(y);
    return r;
  }
  */

  template<typename TYPE>
  inline SO3vecD<TYPE> CGproduct(const SO3vecD<TYPE>& x, const SO3vecD<TYPE>& y, const int maxl=-1){
    SO3vecD<TYPE> R=SO3vecD<TYPE>::zero().batch(std::max(x.nbatch(),y.nbatch())).grid(x.gdims())
      .tau(CGproduct(x.tau(),y.tau(),maxl));
    R.add_CGproduct(x,y);
    return R;
  }

}

#endif  
