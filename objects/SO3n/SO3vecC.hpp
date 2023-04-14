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
//#include "TensorView.hpp"
//#include "TensorVirtual.hpp"
#include "Gvec.hpp"
#include "SO3vecView.hpp"


namespace GElib{

  template<typename RTYPE>
  class SO3vecC:public Gvec<SO3vecView<RTYPE> >{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::GstridesB GstridesB;

    typedef cnine::MemArr<complex<RTYPE> > MemArr;
    typedef GvecArray<SO3vecView<RTYPE> > GvecArray;

    using GvecArray::parts;



  public: // ---- Constructors -------------------------------------------------------------------------------


    SO3vecC(){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vecC(const Gdims& _adims, const SO3type& _tau, const FILLTYPE& fill, const int _dev=0){
      for(int l=0; l<_tau.size(); l++){
	Gdims dims({2*l+1,_tau[l]});
	parts[l]=new SO3partArrayView<RTYPE>(_adims,dims,fill,_dev);
      }
    }


  public: // ---- Named constructors -------------------------------------------------------------------------


    static SO3vecC zero(const Gdims& _adims, const SO3type& _tau, const int _dev=0){
      return SO3vecC(_adims,_tau,cnine::fill_zero(),_dev);
    }

    static SO3vecC sequential(const Gdims& _adims, const SO3type& _tau, const int _dev=0){
      return SO3vecC(_adims,_tau,cnine::fill_sequential(),_dev);
    }

    static SO3vecC gaussian(const Gdims& _adims, const SO3type& _tau, const int _dev=0){
      return SO3vecC(_adims,_tau,cnine::fill_gaussian(),_dev);
    }



  };


  // ---- Functions ------------------------------------------------------------------------------------------


  template<typename RTYPE>
  inline SO3vecC<RTYPE> operator+(const SO3vecC<RTYPE>& x, const SO3vecC<RTYPE>& y){
    SO3vecC<RTYPE> r(x);
    r.add(y);
    return r;
  }


}

#endif 
