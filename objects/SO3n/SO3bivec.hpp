// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3bivecC
#define _GElibSO3bivecC

#include "GElib_base.hpp"
#include "Gvec.hpp"
#include "SO3bivecView.hpp"
#include "SO3bitype.hpp"
#include "SO3vecC.hpp"
#include "diff_class.hpp"

// SO3bivec<RTYPE> -> Gvec<SO3bivecView<RTYPE> > -> SO3bivecView<RTYPE> -> 
// BatchedGvecView<int,BatchedSO3partView<RTYPE>,SO3bivecView<RTYPE> > -> GvecView<<int,BatchedSO3partView<RTYPE> > 

namespace GElib{

  template<typename RTYPE>
  class SO3bivec: public Gvec<SO3bivecView<RTYPE> >, 
		public cnine::diff_class<SO3bivec<RTYPE> >{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::GstridesB GstridesB;

    typedef cnine::MemArr<complex<RTYPE> > MemArr;
    typedef Gvec<SO3bivecView<RTYPE> > _Gvec;
    typedef cnine::diff_class<SO3bivec<RTYPE> > diff_class; 

    using _Gvec::_Gvec;
    using _Gvec::parts;

#ifdef WITH_FAKE_GRAD
    using diff_class::grad; 
    using diff_class::add_to_grad; 
#endif

    ~SO3bivec(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    SO3bivec(){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3bivec(const int _b, const SO3bitype& _tau, const FILLTYPE& fill, const int _dev=0){
      for(auto p:_tau){
	int l1=p.first.first;
	int l2=p.first.second;
	Gdims dims({2*l1+1,2*l2+1,p.second});
	parts[p.first]=new SO3bipartView<RTYPE>(_b,dims,fill,_dev);
      }
    }


  public: // ---- Named constructors -------------------------------------------------------------------------


    static SO3bivec zero(const int _b, const SO3bitype& _tau, const int _dev=0){
      return SO3bivec(_b,_tau,cnine::fill_zero(),_dev);
    }

    static SO3bivec sequential(const int _b, const SO3bitype& _tau, const int _dev=0){
      return SO3bivec(_b,_tau,cnine::fill_sequential(),_dev);
    }

    static SO3bivec gaussian(const int _b, const SO3bitype& _tau, const int _dev=0){
      return SO3bivec(_b,_tau,cnine::fill_gaussian(),_dev);
    }


    static SO3bivec* new_zeros_like(const SO3bivec& x){
      return new SO3bivec(x.getb(),x.get_tau(),cnine::fill_zero(),x.device());
    }

  public: // ---- Static

  public: // ---- ATen --------------------------------------------------------------------------------------

    
    #ifdef _WITH_ATEN

    //SO3bivec(const vector<at::Tensor>& v){
    //for(auto& p:v)
    //parts[(p.size(1)-1)/2]=new SO3partView<RTYPE>(p);
    //}

    #endif 



  };


  // ---- Functions ------------------------------------------------------------------------------------------


  template<typename RTYPE>
  inline SO3bivec<RTYPE> operator+(const SO3bivec<RTYPE>& x, const SO3bivec<RTYPE>& y){
    SO3bivec<RTYPE> r(x);
    r.add(y);
    return r;
  }

  template<typename TYPE>
  inline SO3vec<TYPE> CGtransform(const SO3bivecView<TYPE>& x, const int maxl=-1){
    SO3vec<TYPE> R=SO3vec<TYPE>::zero(x.getb(),GElib::CGtransform(x.get_tau(),maxl),x.device());
    x.add_CGtransform_to(R);
    return R;
  }


}

#endif 
