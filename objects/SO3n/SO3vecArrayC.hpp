// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3vecArray
#define _GElibSO3vecArray

#include "GElib_base.hpp"
//#include "TensorView.hpp"
#include "TensorVirtual.hpp"
#include "GvecArray.hpp"
#include "SO3vecArrayView.hpp"


namespace GElib{

  template<typename RTYPE>
  class SO3vecArray:public GvecArray<SO3vecArrayView<RTYPE> >{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::GstridesB GstridesB;

    typedef cnine::MemArr<complex<RTYPE> > MemArr;
    typedef GvecArray<SO3vecArrayView<RTYPE> > GvecArray;

    using GvecArray::parts;



  public: // ---- Constructors -------------------------------------------------------------------------------


    SO3vecArray(){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vecArray(const int b, const Gdims& _adims, const SO3type& _tau, const FILLTYPE& fill, const int _dev=0){
      for(int l=0; l<_tau.size(); l++){
	Gdims dims({2*l+1,_tau[l]});
	parts[l]=new SO3partArrayView<RTYPE>(b,_adims,dims,fill,_dev);
      }
    }


  public: // ---- Named constructors -------------------------------------------------------------------------


    static SO3vecArray zero(const int b, const Gdims& _adims, const SO3type& _tau, const int _dev=0){
      return SO3vecArray(b,_adims,_tau,cnine::fill_zero(),_dev);
    }

    static SO3vecArray sequential(const int b, const Gdims& _adims, const SO3type& _tau, const int _dev=0){
      return SO3vecArray(b,_adims,_tau,cnine::fill_sequential(),_dev);
    }

    static SO3vecArray gaussian(const int b, const Gdims& _adims, const SO3type& _tau, const int _dev=0){
      return SO3vecArray(b,_adims,_tau,cnine::fill_gaussian(),_dev);
    }


  };


  // ---- Functions ------------------------------------------------------------------------------------------


  template<typename RTYPE>
  inline SO3vecArray<RTYPE> operator+(const SO3vecArray<RTYPE>& x, const SO3vecArray<RTYPE>& y){
    SO3vecArray<RTYPE> r(x);
    r.add(y);
    return r;
  }

  template<typename TYPE>
  inline SO3vecArray<TYPE> CGproduct(const SO3vecArrayView<TYPE>& x, const SO3vecArrayView<TYPE>& y, const int maxl=-1){
    GELIB_ASSRT(x.getb()==y.getb());
    GELIB_ASSRT(x.get_adims()==y.get_adims());
    SO3vecArray<TYPE> R=SO3vecArray<TYPE>::zero(x.getb(),x.get_adims(),
      GElib::CGproduct(x.get_tau(),y.get_tau(),maxl),x.device());
    R.add_CGproduct(x,y);
    //add_vCGproduct(R,x,y);
    return R;
  }

}

#endif 
