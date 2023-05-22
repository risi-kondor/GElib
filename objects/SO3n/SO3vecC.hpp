// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3vecC
#define _GElibSO3vecC

#include "GElib_base.hpp"
#include "Gvec.hpp"
#include "SO3vecView.hpp"
#include "diff_class.hpp"

// SO3vec<RTYPE> -> Gvec<SO3vecView<RTYPE> > -> SO3vecView<RTYPE> -> 
// BatchedGvecView<int,BatchedSO3partView<RTYPE>,SO3vecView<RTYPE> > -> GvecView<<int,BatchedSO3partView<RTYPE> > 

namespace GElib{

  template<typename RTYPE>
  class SO3vec: public Gvec<SO3vecView<RTYPE> >, 
		public cnine::diff_class<SO3vec<RTYPE> >{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::GstridesB GstridesB;

    typedef cnine::MemArr<complex<RTYPE> > MemArr;
    typedef Gvec<SO3vecView<RTYPE> > _Gvec;
    typedef cnine::diff_class<SO3vec<RTYPE> > diff_class; 

    using _Gvec::_Gvec;
    using _Gvec::parts;

#ifdef WITH_FAKE_GRAD
    using diff_class::grad; 
    using diff_class::add_to_grad; 
#endif

    ~SO3vec(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    SO3vec(){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vec(const int _b, const SO3type& _tau, const FILLTYPE& fill, const int _dev=0){
      for(int l=0; l<_tau.size(); l++){
	Gdims dims({2*l+1,_tau[l]});
	parts[l]=new SO3partView<RTYPE>(_b,dims,fill,_dev);
      }
    }


  public: // ---- Named constructors -------------------------------------------------------------------------


    static SO3vec zero(const int _b, const SO3type& _tau, const int _dev=0){
      return SO3vec(_b,_tau,cnine::fill_zero(),_dev);
    }

    static SO3vec sequential(const int _b, const SO3type& _tau, const int _dev=0){
      return SO3vec(_b,_tau,cnine::fill_sequential(),_dev);
    }

    static SO3vec gaussian(const int _b, const SO3type& _tau, const int _dev=0){
      return SO3vec(_b,_tau,cnine::fill_gaussian(),_dev);
    }


    static SO3vec* new_zeros_like(const SO3vec& x){
      return new SO3vec(x.getb(),x.get_tau(),cnine::fill_zero(),x.device());
    }


  public: // ---- ATen --------------------------------------------------------------------------------------

    
    #ifdef _WITH_ATEN

    SO3vec(const vector<at::Tensor>& v){
      for(auto& p:v)
	parts[(p.size(1)-1)/2]=new SO3partView<RTYPE>(p);
    }

    #endif 



  };


  // ---- Functions ------------------------------------------------------------------------------------------


  template<typename RTYPE>
  inline SO3vec<RTYPE> operator+(const SO3vec<RTYPE>& x, const SO3vec<RTYPE>& y){
    SO3vec<RTYPE> r(x);
    r.add(y);
    return r;
  }

  template<typename TYPE>
  inline SO3vec<TYPE> CGproduct(const SO3vecView<TYPE>& x, const SO3vecView<TYPE>& y, const int maxl=-1){
    GELIB_ASSRT(x.getb()==y.getb());
    SO3vec<TYPE> R=SO3vec<TYPE>::zero(x.getb(),GElib::CGproduct(x.get_tau(),y.get_tau(),maxl),x.device());
    R.add_CGproduct(x,y);
    //add_vCGproduct(R,x,y);
    return R;
  }


}

#endif 
