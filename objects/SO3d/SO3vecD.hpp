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
#include "diff_class.hpp"
#include "SO3group.hpp"
#include "SO3partD.hpp"
#include "GvecD.hpp"


namespace GElib{

  template<typename TYPE>
  class SO3vecD: public GvecD<SO3group,TYPE>, public cnine::diff_class<SO3vecD<TYPE> >{
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


  public: // ---- GvecSpec<SO3group> -------------------------------------------------------------------------------


    SO3vecD(const GvecSpec<SO3group>& spec):
      BASE(spec){
      for(int l=0; l<spec._tau.size(); l++)
	parts[l]=new SO3partD<TYPE>(SO3partSpec(spec).l(l).n(spec._tau(l)));
    }
    
    static GvecSpec<SO3group> raw() {return GvecSpec<SO3group>().raw();}
    static GvecSpec<SO3group> zero() {return GvecSpec<SO3group>().zero();}
    static GvecSpec<SO3group> sequential() {return GvecSpec<SO3group>().sequential();}
    static GvecSpec<SO3group> gaussian() {return GvecSpec<SO3group>().gaussian();}

    GvecSpec<SO3group> spec() const{
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


  template<typename TYPE>
  inline SO3vecD<TYPE> CGproduct(const SO3vecD<TYPE>& x, const SO3vecD<TYPE>& y, const int maxl=-1){
    SO3vecD<TYPE> R=SO3vecD<TYPE>::zero().batch(std::max(x.nbatch(),y.nbatch())).grid(x.gdims())
      .tau(CGproduct(x.tau(),y.tau(),maxl));
    R.add_CGproduct(x,y);
    return R;
  }

  template<typename TYPE>
  inline SO3vecD<TYPE> DiagCGproduct(const SO3vecD<TYPE>& x, const SO3vecD<TYPE>& y, const int maxl=-1){
    SO3vecD<TYPE> R=SO3vecD<TYPE>::zero().batch(std::max(x.nbatch(),y.nbatch())).grid(x.gdims())
      .tau(DiagCGproduct(x.tau(),y.tau(),maxl));
    R.add_DiagCGproduct(x,y);
    return R;
  }

  template<typename TYPE>
  inline SO3vecD<TYPE> Fproduct(const SO3vecD<TYPE>& x, const SO3vecD<TYPE>& y, const int maxl=-1){
    if(maxl==-1) maxl=x.max_irrep()+y.max_irrep();
    SO3vecD<TYPE> R=SO3vecD<TYPE>::zero().batch(std::max(x.nbatch(),y.nbatch())).grid(x.gdims()).fourier(maxl);
    R.add_Fproduct(x,y);
    return R;
  }

  template<typename TYPE>
  inline SO3vecD<TYPE> Fmodsq(const SO3vecD<TYPE>& x, const int maxl=-1){
    if(maxl==-1) maxl=2*x.max_irrep();
    SO3vecD<TYPE> R=SO3vecD<TYPE>::zero().batch(x.nbatch()).grid(x.gdims()).fourier(maxl);
    R.add_Fproduct(x,x.transp(),1);
    return R;
  }

}

#endif  
