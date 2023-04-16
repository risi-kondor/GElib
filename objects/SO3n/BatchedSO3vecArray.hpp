// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibBatchedSO3vecArray
#define _GElibBatchedSO3vecArray

#include "GElib_base.hpp"
//#include "TensorView.hpp"
#include "TensorVirtual.hpp"
#include "GvecArray.hpp"
#include "BatchedSO3vecArrayView.hpp"


namespace GElib{

  template<typename RTYPE>
  class BatchedSO3vecArray:public GvecArray<BatchedSO3vecArrayView<RTYPE> >{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::GstridesB GstridesB;

    typedef cnine::MemArr<complex<RTYPE> > MemArr;
    typedef GvecArray<BatchedSO3vecArrayView<RTYPE> > GvecArray;

    using GvecArray::parts;



  public: // ---- Constructors -------------------------------------------------------------------------------


    BatchedSO3vecArray(){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    BatchedSO3vecArray(const int b, const Gdims& _adims, const SO3type& _tau, const FILLTYPE& fill, const int _dev=0){
      for(int l=0; l<_tau.size(); l++){
	Gdims dims({2*l+1,_tau[l]});
	parts[l]=new BatchedSO3partArrayView<RTYPE>(b,_adims,dims,fill,_dev);
      }
    }


  public: // ---- Named constructors -------------------------------------------------------------------------


    static BatchedSO3vecArray zero(const int b, const Gdims& _adims, const SO3type& _tau, const int _dev=0){
      return BatchedSO3vecArray(b,_adims,_tau,cnine::fill_zero(),_dev);
    }

    static BatchedSO3vecArray sequential(const int b, const Gdims& _adims, const SO3type& _tau, const int _dev=0){
      return BatchedSO3vecArray(b,_adims,_tau,cnine::fill_sequential(),_dev);
    }

    static BatchedSO3vecArray gaussian(const int b, const Gdims& _adims, const SO3type& _tau, const int _dev=0){
      return BatchedSO3vecArray(b,_adims,_tau,cnine::fill_gaussian(),_dev);
    }



  };


  // ---- Functions ------------------------------------------------------------------------------------------


  template<typename RTYPE>
  inline BatchedSO3vecArray<RTYPE> operator+(const BatchedSO3vecArray<RTYPE>& x, const BatchedSO3vecArray<RTYPE>& y){
    BatchedSO3vecArray<RTYPE> r(x);
    r.add(y);
    return r;
  }

  template<typename TYPE>
  inline BatchedSO3vecArray<TYPE> CGproduct(const BatchedSO3vecArrayView<TYPE>& x, const BatchedSO3vecArrayView<TYPE>& y, const int maxl=-1){
    GELIB_ASSRT(x.getb()==y.getb());
    GELIB_ASSRT(x.get_adims()==y.get_adims());
    BatchedSO3vecArray<TYPE> R=BatchedSO3vecArray<TYPE>::zero(x.getb(),x.get_adims(),
      GElib::CGproduct(x.get_tau(),y.get_tau(),maxl),x.device());
    add_vCGproduct(R,x,y);
    return R;
  }

}

#endif 
