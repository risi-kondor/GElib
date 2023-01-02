
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3vec
#define _SO3vec

#include "GElib_base.hpp"
#include "SO3type.hpp"
#include "SO3partB.hpp"
#include "SO3vecB.hpp"
#include "SO3element.hpp"


namespace GElib{


  class SO3vec: public GELIB_SO3VEC_IMPL{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::CoutLock CoutLock;

    typedef cnine::RscalarObj rscalar;
    typedef cnine::CscalarObj cscalar;
    typedef cnine::CtensorObj ctensor;
    typedef cnine::CtensorPackObj ctensorpack;


    SO3vec(){}

    ~SO3vec(){
    }


    // ---- Constructors --------------------------------------------------------------------------------------


    using GELIB_SO3VEC_IMPL::GELIB_SO3VEC_IMPL;


    //SO3vecB(const cnine::fill_noalloc& dummy, const SO3type& _tau, const int _nbu, const int _fmt, const int _dev):
    //tau(_tau), nbu(_nbu), fmt(_fmt), dev(_dev){}



    //template<typename FILLTYPE, typename = typename 
    //     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    //SO3vec(const int b, const SO3type& tau, const FILLTYPE fill, const int _dev):
    //(const int b, const SO3type& tau, const FILLTYPE fill, const int _dev){
    //for(int l=0; l<tau.size(); l++)
    //parts.push_back(new SO3partB(b,l,tau[l],fill,_dev));
    //}

    
    //template<typename FILLTYPE, typename = typename 
    //	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    //SO3vecB(const int b, const int maxl, const FILLTYPE fill, const int _dev){
    //for(int l=0; l<=maxl; l++)
    //parts.push_back(new SO3partB(b,l,2*l+1,fill,_dev));
    //}

    
    // ---- Named constructors --------------------------------------------------------------------------------

    
    static SO3vec raw(const SO3type& tau, const int _dev=0){
      return SO3vec(1,tau,cnine::fill_raw(),_dev);}
    static SO3vec raw(const int b, const SO3type& tau, const int _dev=0){
      return SO3vec(b,tau,cnine::fill_raw(),_dev);}

    static SO3vec zero(const SO3type& tau, const int _dev=0){
      return SO3vec(1,tau,cnine::fill_zero(),_dev);}
    static SO3vec zero(const int b, const SO3type& tau, const int _dev=0){
      return SO3vec(b,tau,cnine::fill_zero(),_dev);}

    static SO3vec gaussian(const SO3type& tau, const int _dev=0){
      return SO3vec(1,tau,cnine::fill_gaussian(),_dev);}
    static SO3vec gaussian(const int b, const SO3type& tau, const int _dev=0){
      return SO3vec(b,tau,cnine::fill_gaussian(),_dev);}
    

    // ---- Fourier constructors -----------------------------------------------------------------------------


    static SO3vec Fzero(const int maxl, const int _dev=0){
      return SO3vec(1,maxl,cnine::fill_zero(),_dev);}
    static SO3vec Fzero(const int b, const int maxl, const int _dev=0){
      return SO3vec(b,maxl,cnine::fill_zero(),_dev);}

    static SO3vec Fraw(const int maxl, const int _dev=0){
      return SO3vec(1,maxl,cnine::fill_raw(),_dev);}
    static SO3vec Fraw(const int b, const int maxl, const int _dev=0){
      return SO3vec(b,maxl,cnine::fill_raw(),_dev);}

    static SO3vec Fgaussian(const int maxl, const int _dev=0){
      return SO3vec(1,maxl,cnine::fill_gaussian(),_dev);}
    static SO3vec Fgaussian(const int b, const int maxl, const int _dev=0){
      return SO3vec(b,maxl,cnine::fill_gaussian(),_dev);}

    
    // ---- Like constructors --------------------------------------------------------------------------------


    static SO3vec raw_like(const SO3vecB& x){
      return SO3vec::raw(x.getb(),x.get_tau(),x.get_dev());}
    static SO3vec zero_like(const SO3vecB& x){
      return SO3vec::zero(x.getb(),x.get_tau(),x.get_dev());}
    static SO3vec gaussian_like(const SO3vecB& x){
      return SO3vec::gaussian(x.getb(),x.get_tau(),x.get_dev());}


    // ---- Copying -------------------------------------------------------------------------------------------


    SO3vec(const SO3vec& x):
      GELIB_SO3VEC_IMPL(x){
      GELIB_COPY_WARNING();
    }

    SO3vec(SO3vec&& x):
      GELIB_SO3VEC_IMPL(std::move(x)){
      GELIB_MOVE_WARNING();
    }


    // ---- Conversions ---------------------------------------------------------------------------------------


    SO3vec(const GELIB_SO3VEC_IMPL& x):
      GELIB_SO3VEC_IMPL(x){
      GELIB_CONVERT_WARNING();
    }

    SO3vec(GELIB_SO3VEC_IMPL&& x):
      GELIB_SO3VEC_IMPL(std::move(x)){
      GELIB_MCONVERT_WARNING();
    }


    // ---- Transport -----------------------------------------------------------------------------------------

    /*
    SO3vecB& move_to_device(const int _dev){
      for(auto p:parts)
	p->move_to_device(_dev);
      return *this;
    }
    
    SO3vecB to_device(const int _dev) const{
      SO3vecB R;
      for(auto p:parts)
	R.parts.push_back(new SO3partB(p->to_device(_dev)));
      return R;
    }
    */

 
    // ---- Access --------------------------------------------------------------------------------------------


    // ---- Operations ---------------------------------------------------------------------------------------


    SO3vec operator-(const SO3vec& y) const{
      return GELIB_SO3VEC_IMPL::operator-(y);
    }


    // ---- Rotations ----------------------------------------------------------------------------------------


    SO3vec rotate(const SO3element& r){
      return GELIB_SO3VEC_IMPL::rotate(r);
    }

    
    SO3vec CGproduct(const SO3vec& y, const int maxl=-1) const{
      return GELIB_SO3VEC_IMPL::CGproduct(y,maxl);
    }


    SO3vecB Fproduct(const SO3vecB& y, int maxl=-1) const{
      return GELIB_SO3VEC_IMPL::Fproduct(y,maxl);
    }


    SO3vecB Fmodsq(int maxl=-1) const{
      return GELIB_SO3VEC_IMPL::Fmodsq(maxl);
    }




  public: // ---- I/O ---------------------------------------------------------------------------------------


    static string classname(){
      return "GElib::SO3vec";
    }

    string repr(const string indent="") const{
      return "<GElib::SO3vec of type"+get_tau().str()+">";
    }
    
    friend ostream& operator<<(ostream& stream, const SO3vec& x){
      stream<<x.str(); return stream;
    }

  };


  // ---- Post-class functions -------------------------------------------------------------------------------


  inline SO3vec CGproduct(const SO3vec& x, const SO3vec& y, const int maxl=-1){
    return x.CGproduct(y,maxl);
  }

  inline SO3vec CGsquare(const SO3vec& x, const int maxl=-1){
    return x.CGsquare(maxl);
  }

  inline SO3vec Fproduct(const SO3vec& x, const SO3vec& y, const int maxl=-1){
    return x.Fproduct(y,maxl);
  }

  inline SO3vec Fmodsq(const SO3vec& x, const int maxl=-1){
    return x.Fmodsq(maxl);
  }

}

#endif
